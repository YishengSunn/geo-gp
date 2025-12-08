import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import torch
import matplotlib.pyplot as plt

class SkyGP_MOE:
    def __init__(
        self,
        x_dim,
        y_dim,
        max_data_per_expert=50,
        nearest_k=1,
        max_experts=40,
        replacement=False,
        # 在线优化配置（可按需调）
        min_points=20,          # 专家首次达到该样本数时触发优化
        batch_step=10,          # 之后每新增这么多样本再微调一次
        window_size=20,         # 窗口化MLE使用的最近样本数
        light_maxiter=20        # 每次超参优化允许的最大迭代步数
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_data = max_data_per_expert
        self.max_experts = max_experts
        self.nearest_k = nearest_k

        # 数据存储
        self.X_list = []            # list of (D, Nmax)
        self.Y_list = []            # list of (P, Nmax)
        self.localCount = []        # list of N (有效样本数)
        self.expert_centers = []    # list of (D,)
        self.drop_centers = []      # list of (D,)
        self.drop_counts = []       # list of int
        self.model_params = {}      # dict: expert_id -> dict of log params
        self.L_all = []             # list of (Nmax, Nmax)
        self.alpha_all = []         # list of (Nmax, P)
        self.pretrained_params = None

        # 维护信息
        self.REBUILD_EVERY_N = 1
        self.replace_since_update = []
        self.last_sorted_experts = None
        self.last_prediction_cache = {}
        self.expert_usage_counts = []  # 每个专家被用于预测/查询的次数
        self.replacement = replacement

        # 在线优化配置
        self.MIN_POINTS = int(min_points)
        self.BATCH_STEP = int(batch_step)
        self.WINDOW_SIZE = int(window_size) if window_size is not None else None
        self.LIGHT_MAXITER = int(light_maxiter)
        
        # ===== Shared hyperparameters =====
        self.global_params = None   # {'log_sigma_f', 'log_sigma_n', 'log_lengthscale'}
        self._since_global_opt = 0  # 新增样本计数（触发全局优化）

    # ---------------------------
    # 内核与初始化
    # ---------------------------
    def kernel_np(self, X1, X2, lengthscale, sigma_f):
        # X1, X2: (D, N), lengthscale: (D,), sigma_f: scalar
        X1_scaled = X1 / lengthscale[:, None]
        X2_scaled = X2 / lengthscale[:, None]
        # 两两平方距离
        A = np.sum(X1_scaled**2, axis=0, keepdims=True)
        B = np.sum(X2_scaled**2, axis=0, keepdims=True)
        dists = A.T + B - 2.0 * (X1_scaled.T @ X2_scaled)
        dists[dists < 0] = 0.0
        return (sigma_f**2) * np.exp(-0.5 * dists)

    def init_model_params(self, model_id, pretrained_params=None):
        # 允许传入预训练参数，否则默认初始化
        if pretrained_params:
            self.pretrained_params = pretrained_params
            outputscale, noise, lengthscale = pretrained_params
            log_sigma_f = np.log(outputscale.flatten())
            log_sigma_n = np.log(noise.flatten())
            if lengthscale.ndim == 2 and lengthscale.shape[1] == self.y_dim:
                log_lengthscale = np.log(lengthscale)
            else:
                log_lengthscale = np.log(lengthscale.squeeze())
        elif self.pretrained_params:
            outputscale, noise, lengthscale = self.pretrained_params
            log_sigma_f = np.log(outputscale.flatten())
            log_sigma_n = np.log(noise.flatten())
            if lengthscale.ndim == 2 and lengthscale.shape[1] == self.y_dim:
                log_lengthscale = np.log(lengthscale)
            else:
                log_lengthscale = np.log(lengthscale.squeeze())
        else:
            log_sigma_f = np.log(np.ones(self.y_dim))
            log_sigma_n = np.log(np.ones(self.y_dim) * 0.01)
            if self.y_dim == 1:
                log_lengthscale = np.log(np.ones((self.x_dim,)))
            else:
                log_lengthscale = np.log(np.ones((self.x_dim, self.y_dim)))

        self.model_params[model_id] = {
            'log_sigma_f': log_sigma_f,
            'log_sigma_n': log_sigma_n,
            'log_lengthscale': log_lengthscale
        }

    def get_hyperparams(self):
        params = self.global_params
        sigma_f = np.exp(params['log_sigma_f'])
        sigma_n = np.exp(params['log_sigma_n'])
        lengthscale = np.exp(params['log_lengthscale'])
        return {
            'sigma_f': sigma_f,
            'sigma_n': sigma_n,
            'lengthscale': lengthscale
        }

    # ---------------------------
    # 专家管理（创建/替换/插入）
    # ---------------------------
    def _create_new_expert(self, model, src_params=None):
        self.X_list.append(np.zeros((self.x_dim, self.max_data)))
        self.Y_list.append(np.zeros((self.y_dim, self.max_data)))
        self.localCount.append(0)
        self.expert_centers.append(np.zeros(self.x_dim))
        self.drop_centers.append(np.zeros(self.x_dim))
        self.drop_counts.append(0)
        self.L_all.append(np.zeros((self.max_data, self.max_data)))
        self.alpha_all.append(np.zeros((self.max_data, self.y_dim)))
        self.expert_usage_counts.append(0)
        self.replace_since_update.append(0)

        if src_params is not None:
            self.model_params[model] = {
                'log_sigma_f': src_params['log_sigma_f'].copy(),
                'log_sigma_n': src_params['log_sigma_n'].copy(),
                'log_lengthscale': src_params['log_lengthscale'].copy(),
            }
        else:
            self.init_model_params(model)

    def _replace_expert(self, idx, inherit_idx=None):
        # 清空槽位
        self.X_list[idx] = np.zeros((self.x_dim, self.max_data))
        self.Y_list[idx] = np.zeros((self.y_dim, self.max_data))
        self.localCount[idx] = 0
        self.expert_centers[idx] = np.zeros(self.x_dim)
        self.drop_centers[idx] = np.zeros(self.x_dim)
        self.drop_counts[idx] = 0
        self.L_all[idx] = np.zeros((self.max_data, self.max_data))
        self.alpha_all[idx] = np.zeros((self.max_data, self.y_dim))
        self.replace_since_update[idx] = 0
        self.expert_usage_counts[idx] = 0

        # 继承最近已有专家的参数
        if inherit_idx is not None:
            self.model_params[idx] = {
                'log_sigma_f': self.model_params[inherit_idx]['log_sigma_f'].copy(),
                'log_sigma_n': self.model_params[inherit_idx]['log_sigma_n'].copy(),
                'log_lengthscale': self.model_params[inherit_idx]['log_lengthscale'].copy(),
            }
        else:
            self.init_model_params(idx)

    def _insert_into_expert(self, model, x, y):
        idx = self.localCount[model]
        self.X_list[model][:, idx] = x
        self.Y_list[model][:, idx] = y
        self.localCount[model] += 1
        # 更新中心
        self.expert_centers[model] = (x if idx == 0 else (self.expert_centers[model] * idx + x) / (idx + 1))
        # 增量更新 L/alpha
        self.update_param_incremental(x, y, model)

        # 在线优化触发（只优化该专家，不共享）
        # —— 改为“全局触发”：累计一定数量新样本后做一次全局共享超参优化 ——
        self._since_global_opt += 1
        total_pts = sum(self.localCount)
        if (total_pts >= self.MIN_POINTS) and (self._since_global_opt % self.BATCH_STEP == 0):
            try:
                ok = self.optimize_hyperparams_global(
                    max_iter=self.LIGHT_MAXITER,
                    verbose=True,
                    window_size=self.WINDOW_SIZE,
                    adam_lr=1e-3   # 可用你的 params["adam_lr"]
                )
                if ok:
                    self._since_global_opt = 0  # 重置计数
            except Exception as e:
                print(f"[opt-online-global] 失败: {e}")

    # ---------------------------
    # 增量/重建 & 预测
    # ---------------------------
    def update_param(self, model):
        if self.localCount[model] == 0:
            return
        p = 0
        idx = self.localCount[model]
        # >>> 改这里：用共享 or 私有log超参
        log_sigma_f, log_sigma_n, log_lengthscale = self._get_param_logs(model, p)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        lengthscale = np.exp(log_lengthscale)

        X_subset = self.X_list[model][:, :idx]
        Y_subset = self.Y_list[model][:idx, :]  # (idx, P)

        # 每个输出维度单独做 alpha
        K = self.kernel_np(X_subset, X_subset, lengthscale, sigma_f)
        K[np.diag_indices_from(K)] += (sigma_n ** 2)
        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(idx))
        except np.linalg.LinAlgError:
            # 兜底
            L = np.linalg.cholesky(K + 1e-4 * np.eye(idx))

        self.L_all[model][:idx, :idx] = L
        for p in range(self.y_dim):
            y_p = self.Y_list[model][p, :idx]
            aux_alpha = solve_triangular(L, y_p, lower=True)
            self.alpha_all[model][:idx, p] = solve_triangular(L.T, aux_alpha, lower=False)

    def update_param_incremental(self, x, y, model):
        p = 0
        idx = self.localCount[model]
        if idx == 0:
            return

        # >>> 改这里
        log_sigma_f, log_sigma_n, log_lengthscale = self._get_param_logs(model, p)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        lengthscale = np.exp(log_lengthscale)

        if idx == 1:
            # 第一条数据时，直接构建 1x1
            kxx = self.kernel_np(x[:, None], x[:, None], lengthscale, sigma_f)[0, 0] + sigma_n**2
            L = np.sqrt(kxx)
            self.L_all[model][0, 0] = L
            self.alpha_all[model][0, p] = y[p] / (L * L)
            return

        # 之前的样本
        X_prev = self.X_list[model][:, :idx - 1]
        # y_vals 只用于构建 alpha
        y_vals = self.Y_list[model][p, :idx]

        # 尝试使用缓存（若来自最近一次 predict）
        cached = self.last_prediction_cache.get(model, {}).get(p, None)
        if cached is not None and cached['k_star'].shape[0] == idx - 1:
            b = cached['k_star']
            Lx = cached['v']
        else:
            b = self.kernel_np(X_prev, x[:, None], lengthscale, sigma_f).flatten()
            L_prev = self.L_all[model][:idx - 1, :idx - 1]
            Lx = solve_triangular(L_prev, b, lower=True)

        c = self.kernel_np(x[:, None], x[:, None], lengthscale, sigma_f)[0, 0] + sigma_n ** 2
        Lii = np.sqrt(max(c - np.dot(Lx, Lx), 1e-9))

        # 写入增量的 L（下三角）
        self.L_all[model][:idx - 1, idx - 1] = 0.0
        self.L_all[model][idx - 1, :idx - 1] = Lx
        self.L_all[model][idx - 1, idx - 1] = Lii

        # 重新解一次 alpha（复杂度 O(n^2)）
        L_now = self.L_all[model][:idx, :idx]
        aux_alpha = solve_triangular(L_now, y_vals, lower=True)
        self.alpha_all[model][:idx, p] = solve_triangular(L_now.T, aux_alpha, lower=False)

    def predict(self, x_query):
        """选择最近的 nearest_k 个专家做 MOE 聚合"""
        self.last_prediction_cache.clear()

        if len(self.expert_centers) == 0:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        # 根据与专家中心的距离排序
        dists = [(np.linalg.norm(x_query - center), i) for i, center in enumerate(self.expert_centers)]
        dists.sort()
        self.last_sorted_experts = [i for _, i in dists]

        selected = self.last_sorted_experts[:self.nearest_k]
        if not selected:
            return np.zeros(self.y_dim), np.ones(self.y_dim) * 10.0

        mus, vars_ = [], []
        for idx in selected:
            # print("Using expert", idx)
            self.expert_usage_counts[idx] += 1

            n_valid = self.localCount[idx]
            if n_valid == 0:
                # 没数据的专家，给高方差
                mus.append(np.zeros(self.y_dim))
                vars_.append(np.ones(self.y_dim) * 1e6)
                continue

            L = self.L_all[idx][:n_valid, :n_valid]
            alpha = self.alpha_all[idx][:n_valid, :]
            X_snapshot = self.X_list[idx][:, :n_valid]

            mu = np.zeros(self.y_dim)
            var = np.zeros(self.y_dim)

            for p in range(self.y_dim):
                log_sigma_f, log_sigma_n, log_lengthscale = self._get_param_logs(idx, p)
                sigma_f = np.exp(log_sigma_f)
                # sigma_n 当前仅用于需要时
                lengthscale = np.exp(log_lengthscale)

                k_star = self.kernel_np(X_snapshot, x_query[:, None], lengthscale, sigma_f).flatten()
                k_xx = sigma_f ** 2

                mu[p] = np.dot(k_star, alpha[:, p])
                v = solve_triangular(L, k_star, lower=True)
                var[p] = max(k_xx - np.dot(v, v), 1e-6)

                # 仅缓存本专家本输出的中间项，供增量更新复用
                if idx not in self.last_prediction_cache:
                    self.last_prediction_cache[idx] = {}
                self.last_prediction_cache[idx][p] = {
                    'k_star': k_star.copy(),
                    'v': v.copy(),
                    'mu_part': mu[p]
                }

            mus.append(mu)
            vars_.append(var)

        # mus = np.stack(mus)     # (k, P)
        # vars_ = np.stack(vars_) # (k, P)
        # inv_vars = 1.0 / (vars_ + 1e-9)
        # weights = inv_vars / np.sum(inv_vars, axis=0, keepdims=True)

        # mu_moe = np.sum(weights * mus, axis=0)
        # var_moe = np.sum(weights * vars_, axis=0)
        # var_moe = np.clip(var_moe, 1e-6, 1e6)
        # return mu_moe, var_moe
        mus = np.stack(mus)
        vars_ = np.stack(vars_)
        inv_vars = 1.0 / (vars_ + 1e-9)
        sigma0_sq = np.exp(self.model_params[selected[0]]['log_sigma_f'][0])**2
        mu_weighted = np.sum(inv_vars * mus, axis=0)
        denom = np.sum(inv_vars, axis=0)
        mu_corr = mu_weighted
        denom_corr = denom - (len(mus) - 1) / sigma0_sq
        mu_bcm = mu_corr / (denom_corr + 1e-9)
        var_bcm = 1.0 / (denom_corr + 1e-9)
        return mu_bcm, var_bcm

    # ---------------------------
    # 数据写入（含创建/继承/替换逻辑）
    # ---------------------------
    def add_point(self, x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)

        x_uncat = x.copy()
        y_uncat = y.copy()

        # 若已有排序缓存则用；否则按距离计算一次
        expert_order = self.last_sorted_experts if self.last_sorted_experts is not None else []
        if self.last_sorted_experts is None:
            dists = [(np.linalg.norm(x_uncat - center), i) for i, center in enumerate(self.expert_centers)]
            while len(dists) < self.max_experts:
                dists.append((float('inf'), len(dists)))
            dists.sort()
            expert_order = [i for _, i in dists]

        # 遍历最近的 nearest_k 个“槽位编号”
        for model in expert_order[:self.nearest_k]:
            # 槽位不存在 -> 新建（继承 expert_order 中最近的已有专家参数）
            if model >= len(self.X_list):
                if len(self.X_list) < self.max_experts:
                    inherit_idx = next(
                        (i for i in expert_order if i < len(self.X_list) and self.localCount[i] > 0),
                        None
                    )
                    src_params = self.model_params[inherit_idx] if inherit_idx is not None else None
                    self._create_new_expert(model, src_params=src_params)
                else:
                    # 淘汰最少使用专家，并从 expert_order 中最近的已有专家继承
                    least_used_idx = np.argmin(self.expert_usage_counts)
                    inherit_idx = next(
                        (i for i in expert_order if i < len(self.X_list) and i != least_used_idx and self.localCount[i] > 0),
                        None
                    )
                    self._replace_expert(least_used_idx, inherit_idx=inherit_idx)
                    model = least_used_idx

            # 有空位 -> 直接写入
            if self.localCount[model] < self.max_data:
                self._insert_into_expert(model, x_uncat, y_uncat)
                return

            # 满了但不启用替换 -> 看下一个最近专家
            if not self.replacement:
                continue

            # 启用替换逻辑（与你原逻辑一致）
            if self.drop_counts[model] == 0:
                # 第一次丢弃（类似 MASGP）
                d_center = np.linalg.norm(self.expert_centers[model] - x_uncat)
                stored = self.X_list[model][:, :self.max_data]
                dists = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                max_idx = np.argmax(dists)
                if d_center < dists[max_idx]:
                    x_old = self.X_list[model][:, max_idx].copy()
                    y_old = self.Y_list[model][:, max_idx].copy()
                    self.X_list[model][:, max_idx] = x_uncat
                    self.Y_list[model][:, max_idx] = y_uncat
                    # 更新中心
                    self.expert_centers[model] += (x_uncat - x_old) / self.max_data
                    self.drop_centers[model] = x_old
                    self.drop_counts[model] += 1
                    x_uncat, y_uncat = x_old, y_old
                    self.replace_since_update[model] += 1
                    if self.replace_since_update[model] >= self.REBUILD_EVERY_N:
                        self.update_param(model)
                        self.replace_since_update[model] = 0
                    return
            else:
                # 后续丢弃：基于 (keep - drop) 差值最大者
                stored = self.X_list[model][:, :self.max_data]
                d_keep = np.linalg.norm(stored - self.expert_centers[model][:, None], axis=0)
                d_drop = np.linalg.norm(stored - self.drop_centers[model][:, None], axis=0)
                d_new_keep = np.linalg.norm(x_uncat - self.expert_centers[model])
                d_new_drop = np.linalg.norm(x_uncat - self.drop_centers[model])
                d_diff = np.concatenate([(d_keep - d_drop), [d_new_keep - d_new_drop]])
                drop_idx = np.argmax(d_diff)
                if drop_idx < self.max_data:
                    x_old = self.X_list[model][:, drop_idx].copy()
                    y_old = self.Y_list[model][:, drop_idx].copy()
                    self.X_list[model][:, drop_idx] = x_uncat
                    self.Y_list[model][:, drop_idx] = y_uncat
                    self.expert_centers[model] += (x_uncat - x_old) / self.max_data
                    # 更新 drop center（均值）
                    self.drop_centers[model] = (
                        self.drop_centers[model] * self.drop_counts[model] + x_old
                    ) / (self.drop_counts[model] + 1)
                    self.drop_counts[model] += 1
                    x_uncat, y_uncat = x_old, y_old
                    self.replace_since_update[model] += 1
                    if self.replace_since_update[model] >= self.REBUILD_EVERY_N:
                        self.update_param(model)
                        self.replace_since_update[model] = 0
                    return

        # 如果前面的最近专家都无法插入，则新建或替换
        if len(self.X_list) < self.max_experts:
            model = len(self.X_list)
            inherit_idx = next(
                (i for i in expert_order if i < len(self.X_list) and self.localCount[i] > 0),
                None
            )
            src_params = self.model_params[inherit_idx] if inherit_idx is not None else None
            self._create_new_expert(model, src_params=src_params)
        else:
            model = np.argmin(self.expert_usage_counts)
            inherit_idx = next(
                (i for i in expert_order if i < len(self.X_list) and i != model and self.localCount[i] > 0),
                None
            )
            self._replace_expert(model, inherit_idx=inherit_idx)

        self._insert_into_expert(model, x_uncat, y_uncat)

    # ---------------------------
    # MLL 与梯度（用于超参优化）
    # ---------------------------
    def _pairwise_sq_dists_scaled(self, X, lengthscale):
        # X: (D, N); lengthscale: (D,)
        Xs = X / lengthscale[:, None]
        a2 = np.sum(Xs**2, axis=0, keepdims=True)  # (1, N)
        sqd = a2.T + a2 - 2.0 * (Xs.T @ Xs)        # (N, N)
        sqd[sqd < 0] = 0.0
        return sqd

    def _build_K(self, X, log_sigma_f, log_sigma_n, log_lengthscale):
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        lengthscale = np.exp(log_lengthscale)
        sqd = self._pairwise_sq_dists_scaled(X, lengthscale)  # (N, N)
        K_rbf = (sigma_f**2) * np.exp(-0.5 * sqd)
        K = K_rbf.copy()
        n = X.shape[1]
        K[np.diag_indices(n)] += (sigma_n**2)
        return K, K_rbf, sqd, sigma_f, sigma_n, lengthscale

    def _mll_and_grad_for_output(self, X, y, log_sigma_f, log_sigma_n, log_lengthscale, jitter=1e-6):
        """
        X: (D, m), y: (m,)
        返回 (负对数边际似然, 负梯度) 以便最小化
        """
        K, K_rbf, sqd, sigma_f, sigma_n, lengthscale = self._build_K(
            X, log_sigma_f, log_sigma_n, log_lengthscale
        )
        m = X.shape[1]

        # Cholesky
        try:
            L = np.linalg.cholesky(K + jitter * np.eye(m))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-4 * np.eye(m))

        # α = K^{-1} y
        v = solve_triangular(L, y, lower=True)
        alpha = solve_triangular(L.T, v, lower=False)

        # mll
        mll = -0.5 * np.dot(y, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * m * np.log(2 * np.pi)

        # A = αα^T - K^{-1}
        I = np.eye(m)
        Z = solve_triangular(L, I, lower=True)
        Kinv = Z.T @ Z
        A = np.outer(alpha, alpha) - Kinv

        # dK/d log_sigma_f = 2*K_rbf
        dK_dlogsf = 2.0 * K_rbf
        # dK/d log_sigma_n = 2*σ_n^2 * I
        dK_dlogsn = 2.0 * (sigma_n**2) * I

        # dK/d log_lengthscale_d = K_rbf * q_d
        grads_len = []
        for d in range(X.shape[0]):
            xd = (X[d, :] / lengthscale[d]).reshape(1, -1)
            a2 = (xd**2)
            qd = a2.T + a2 - 2.0 * (xd.T @ xd)  # (m, m)
            qd[qd < 0] = 0.0
            dK_dlogld = K_rbf * qd
            grads_len.append(dK_dlogld)

        grad_logsf = 0.5 * np.sum(A * dK_dlogsf)
        grad_logsn = 0.5 * np.sum(A * dK_dlogsn)
        grad_logl = np.array([0.5 * np.sum(A * g) for g in grads_len])

        neg_mll = -mll
        neg_grads = np.concatenate([[-grad_logsf], [-grad_logsn], -grad_logl])
        return neg_mll, neg_grads

    # ---------------------------
    # 单专家在线超参优化（窗口化）
    # ---------------------------
    def optimize_hyperparams(
        self,
        expert_idx,
        p=0,
        max_iter=60,          # 这里当成 Adam 的步数 steps
        verbose=False,
        window_size=None,
        adam_lr=0.00001,         # ✅ Adam 学习率
        weight_decay=0.0,     # 可选 L2 正则
        jitter=1e-6           # 数值稳定的抖动
    ):
        """
        仅对 expert_idx 的第 p 个输出维度做窗口化 MLE，用 PyTorch Adam 优化 log-超参。
        成功后回填到 numpy，并重建该专家的 Cholesky（update_param）。
        """
        # 添加到函数开始处
        losses = []
        
        n = self.localCount[expert_idx]
        if n < 3:
            if verbose:
                print(f"[opt-Adam] expert {expert_idx} n={n} 太少，跳过")
            return False

        # ------- 取最近 window_size 个样本 -------
        if window_size is not None and n > window_size:
            start = n - window_size
        else:
            start = 0
        X_np = self.X_list[expert_idx][:, start:n]            # (D, m)
        y_np = self.Y_list[expert_idx][p, start:n].copy()     # (m,)
        D, m = X_np.shape

        # ------- 取当前超参作为初始化 -------
        params = self.model_params[expert_idx]
        log_sigma_f_np = params['log_sigma_f'][p]
        log_sigma_n_np = params['log_sigma_n'][p]
        if self.y_dim == 1:
            log_lengthscale_np = params['log_lengthscale'].copy()        # (D,)
        else:
            log_lengthscale_np = params['log_lengthscale'][:, p].copy()  # (D,)

        # ------- 转 torch（双精度，更稳） -------
        device = torch.device("cpu")
        X = torch.tensor(X_np, dtype=torch.float64, device=device)       # (D, m)
        y = torch.tensor(y_np, dtype=torch.float64, device=device)       # (m,)

        log_sigma_f = torch.nn.Parameter(torch.tensor(log_sigma_f_np, dtype=torch.float64, device=device))
        log_sigma_n = torch.nn.Parameter(torch.tensor(log_sigma_n_np, dtype=torch.float64, device=device))
        log_lengthscale = torch.nn.Parameter(torch.tensor(log_lengthscale_np, dtype=torch.float64, device=device))

        opt = torch.optim.Adam(
            [{"params": [log_sigma_f, log_sigma_n, log_lengthscale], "lr": adam_lr, "weight_decay": weight_decay}]
        )

        two_pi = torch.tensor(2.0 * np.pi, dtype=torch.float64, device=device)

        def build_kernel():
            # lengthscale: (D,)
            ls = torch.exp(log_lengthscale)
            sf2 = torch.exp(2.0 * log_sigma_f)    # σ_f^2
            sn2 = torch.exp(2.0 * log_sigma_n)    # σ_n^2

            Xs = X / ls.view(-1, 1)               # (D, m)
            a2 = torch.sum(Xs**2, dim=0, keepdim=True)   # (1, m)
            sqd = a2.T + a2 - 2.0 * (Xs.T @ Xs)          # (m, m)
            sqd = sqd.clamp_min(0.0)

            K_rbf = sf2 * torch.exp(-0.5 * sqd)
            K = K_rbf.clone()
            K.view(-1)[::m + 1] += sn2            # 对角加噪声
            return K

        success = False
        for step in range(int(max_iter)):
            opt.zero_grad()
            K = build_kernel()
            try:
                L = torch.linalg.cholesky(K + jitter * torch.eye(m, dtype=torch.float64, device=device))
            except RuntimeError:
                L = torch.linalg.cholesky(K + (1e-4) * torch.eye(m, dtype=torch.float64, device=device))

            v = torch.cholesky_solve(y.view(-1, 1), L)
            alpha = v.view(-1)
            mll = -0.5 * (y @ alpha) - torch.sum(torch.log(torch.diag(L))) - 0.5 * m * torch.log(two_pi)
            loss = -mll
            loss.backward()

            with torch.no_grad():
                log_sigma_n.clamp_(min=np.log(1e-5), max=np.log(10.0))
                log_sigma_f.clamp_(min=np.log(1e-5), max=np.log(10.0))
                log_lengthscale.clamp_(min=np.log(1e-3), max=np.log(1e3))

            opt.step()

            losses.append(loss.item())  # ✅ 收集每一步的 loss

            if verbose and (step % 10 == 0 or step == max_iter - 1):
                print(f"[opt-Adam] expert {expert_idx}, out {p}, step {step:03d}, mll={mll.item():.4f}, lr={adam_lr}")

        # ------- loss 可视化 --------
        if verbose:
            plt.figure(figsize=(6, 3))
            plt.plot(losses, label=f"Expert {expert_idx}, Output {p}")
            plt.xlabel("Step"); plt.ylabel("Negative MLL (loss)")
            plt.title("Hyperparameter Optimization Loss")
            plt.grid(True); plt.legend()
            plt.tight_layout()
            plt.show()

        success = True

        # ------- 回填到 numpy -------
        params['log_sigma_f'][p] = float(log_sigma_f.detach().cpu().numpy())
        params['log_sigma_n'][p] = float(log_sigma_n.detach().cpu().numpy())
        if self.y_dim == 1:
            params['log_lengthscale'] = log_lengthscale.detach().cpu().numpy().astype(np.float64)
        else:
            params['log_lengthscale'][:, p] = log_lengthscale.detach().cpu().numpy().astype(np.float64)

        # 超参变化 => 重建该专家的 Cholesky
        self.update_param(expert_idx)
        print(f"[opt-Adam] expert {expert_idx} output {p} 优化完成")
        # 预测缓存失效
        self.last_prediction_cache.clear()

        return success

    def init_global_params(self, pretrained_params=None):
            """
            初始化全局共享超参（log 形式）。
            - y_dim==1: log_lengthscale.shape = (D,)
            - y_dim>1 : log_lengthscale.shape = (D, P)  (每个输出一套 lengthscale)
            """
            if pretrained_params:
                outputscale, noise, lengthscale = pretrained_params
                log_sigma_f = np.log(outputscale.reshape(-1))
                log_sigma_n = np.log(noise.reshape(-1))
                if self.y_dim == 1:
                    log_lengthscale = np.log(lengthscale.reshape(self.x_dim))
                else:
                    # 允许传入 (D,P) 或 (D,)；不匹配则广播
                    L = np.asarray(lengthscale)
                    if L.ndim == 1:  # (D,)
                        L = np.tile(L[:, None], (1, self.y_dim))
                    log_lengthscale = np.log(L)
            else:
                log_sigma_f = np.log(np.ones(self.y_dim))
                log_sigma_n = np.log(np.ones(self.y_dim) * 0.01)
                if self.y_dim == 1:
                    log_lengthscale = np.log(np.ones((self.x_dim,)))
                else:
                    log_lengthscale = np.log(np.ones((self.x_dim, self.y_dim)))

            self.global_params = {
                'log_sigma_f': log_sigma_f.astype(np.float64),
                'log_sigma_n': log_sigma_n.astype(np.float64),
                'log_lengthscale': log_lengthscale.astype(np.float64),
            }

    def _get_param_logs(self, expert_idx, p):
        """
        统一入口: 拿到“log 参数”。若有全局共享，则用全局；否则回退到该专家的私有。
        返回: (log_sigma_f_p, log_sigma_n_p, log_lengthscale_p) 其中 log_lengthscale_p 形状为 (D,)
        """
        if self.global_params is not None:
            lsf = float(self.global_params['log_sigma_f'][p])
            lsn = float(self.global_params['log_sigma_n'][p])
            if self.y_dim == 1:
                ll = self.global_params['log_lengthscale'].copy()  # (D,)
            else:
                ll = self.global_params['log_lengthscale'][:, p].copy()  # (D,)
            # print(f"[get_param_logs] 使用全局超参 for expert {expert_idx}, output {p}")
            # print(f"  log_sigma_f={np.exp(lsf):.4f}, log_sigma_n={np.exp(lsn):.4f}, log_lengthscale={np.exp(ll)}")
            return lsf, lsn, ll
        else:
            params = self.model_params[expert_idx]
            lsf = float(params['log_sigma_f'][p])
            lsn = float(params['log_sigma_n'][p])
            if self.y_dim == 1:
                ll = params['log_lengthscale'].copy()
            else:
                ll = params['log_lengthscale'][:, p].copy()
            return lsf, lsn, ll

    def rebuild_all_experts(self):
        """按当前（可能更新后的）超参，重建所有专家的 Cholesky 与 alpha。"""
        for e in range(len(self.X_list)):
            if self.localCount[e] > 0:
                self.update_param(e)
        self.last_prediction_cache.clear()
        
    def optimize_hyperparams_global(
        self,
        max_iter=60,
        verbose=False,
        window_size=None,
        adam_lr=1e-3,
        weight_decay=0.0,
        jitter=1e-6
    ):
        """
        用所有专家的数据（每个专家取最近 window_size 个样本）联合最大化总 MLL，
        训练一组“共享”log 超参，并回填到 self.global_params，然后重建所有专家。
        """
        # 若还没初始化共享超参，先按默认初始化一份
        if self.global_params is None:
            self.init_global_params()

        # 收集各专家窗口
        groups = []  # list of (X_np[D,m], y_np[m], out_index p)
        total_m = 0
        for e in range(len(self.X_list)):
            n = self.localCount[e]
            if n < 3:
                continue
            if window_size is not None and n > window_size:
                start = n - window_size
            else:
                start = 0
            X_np = self.X_list[e][:, start:n]  # (D, m)
            m = X_np.shape[1]
            total_m += m
            for p in range(self.y_dim):
                y_np = self.Y_list[e][p, start:n].copy()  # (m,)
                groups.append((X_np, y_np, p))

        if len(groups) == 0:
            if verbose:
                print("[opt-Adam-global] 有效数据不足，跳过")
            return False

        device = torch.device("cpu")
        # --- 共享 log 超参（torch 参数） ---
        log_sigma_f = torch.nn.Parameter(
            torch.tensor(self.global_params['log_sigma_f'], dtype=torch.float64, device=device)
        )
        log_sigma_n = torch.nn.Parameter(
            torch.tensor(self.global_params['log_sigma_n'], dtype=torch.float64, device=device)
        )
        log_lengthscale = torch.nn.Parameter(
            torch.tensor(self.global_params['log_lengthscale'], dtype=torch.float64, device=device)
        )

        opt = torch.optim.Adam(
            [{"params": [log_sigma_f, log_sigma_n, log_lengthscale], "lr": adam_lr, "weight_decay": weight_decay}]
        )
        two_pi = torch.tensor(2.0 * np.pi, dtype=torch.float64, device=device)

        def mll_one_group(X_np, y_np, out_p):
            X = torch.tensor(X_np, dtype=torch.float64, device=device)   # (D, m)
            y = torch.tensor(y_np, dtype=torch.float64, device=device)   # (m,)
            m = X.shape[1]
            if self.y_dim == 1:
                ls = torch.exp(log_lengthscale)                 # (D,)
            else:
                ls = torch.exp(log_lengthscale[:, out_p])       # (D,)
            sf2 = torch.exp(2.0 * log_sigma_f[out_p])
            sn2 = torch.exp(2.0 * log_sigma_n[out_p])

            Xs = X / ls.view(-1, 1)
            a2 = torch.sum(Xs**2, dim=0, keepdim=True)
            sqd = a2.T + a2 - 2.0 * (Xs.T @ Xs)
            sqd = sqd.clamp_min(0.0)

            K_rbf = sf2 * torch.exp(-0.5 * sqd)
            K = K_rbf.clone()
            K.view(-1)[::m + 1] += sn2

            try:
                L = torch.linalg.cholesky(K + jitter * torch.eye(m, dtype=torch.float64, device=device))
            except RuntimeError:
                L = torch.linalg.cholesky(K + (1e-4) * torch.eye(m, dtype=torch.float64, device=device))

            v = torch.cholesky_solve(y.view(-1, 1), L)
            alpha = v.view(-1)
            mll = -0.5 * (y @ alpha) - torch.sum(torch.log(torch.diag(L))) - 0.5 * m * torch.log(two_pi)
            return mll

        losses = []
        for step in range(int(max_iter)):
            opt.zero_grad()
            total_mll = 0.0
            for (X_np, y_np, p) in groups:
                total_mll = total_mll + mll_one_group(X_np, y_np, p)
            loss = -total_mll
            loss.backward()

            with torch.no_grad():
                log_sigma_n.clamp_(min=np.log(1e-5), max=np.log(10.0))
                log_sigma_f.clamp_(min=np.log(1e-5), max=np.log(10.0))
                if self.y_dim == 1:
                    log_lengthscale.clamp_(min=np.log(1e-3), max=np.log(1e3))
                else:
                    log_lengthscale.clamp_(min=np.log(1e-3), max=np.log(1e3))

            opt.step()
            losses.append(float(loss.item()))

            if verbose and (step % 10 == 0 or step == max_iter - 1):
                print(f"[opt-Adam-global] step {step:03d}, -MLL={loss.item():.4f}, lr={adam_lr}, groups={len(groups)}")

        # 回填到 numpy
        self.global_params['log_sigma_f'] = log_sigma_f.detach().cpu().numpy().astype(np.float64)
        self.global_params['log_sigma_n'] = log_sigma_n.detach().cpu().numpy().astype(np.float64)
        self.global_params['log_lengthscale'] = log_lengthscale.detach().cpu().numpy().astype(np.float64)

        # 用新超参重建所有专家
        self.rebuild_all_experts()

        if verbose:
            plt.figure(figsize=(6, 3))
            plt.plot(losses)
            plt.xlabel("Step"); plt.ylabel("Negative total MLL")
            plt.title("Global Hyperparameter Optimization")
            plt.grid(True); plt.tight_layout(); plt.show()

        return True