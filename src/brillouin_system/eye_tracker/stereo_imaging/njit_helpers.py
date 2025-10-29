import numpy as np
from numba import njit



@njit(cache=True, fastmath=True, nogil=True)
def _project_opencv_numba(K, dist, R, t, X):
    """
    X: (3,) in LEFT/world. Returns pixel (u,v). Dist: k1,k2,p1,p2,k3; if shorter, missing terms are 0.
    """
    # camera coords
    Xc0 = R[0,0]*X[0] + R[0,1]*X[1] + R[0,2]*X[2] + t[0]
    Xc1 = R[1,0]*X[0] + R[1,1]*X[1] + R[1,2]*X[2] + t[1]
    Xc2 = R[2,0]*X[0] + R[2,1]*X[1] + R[2,2]*X[2] + t[2]

    # normalized
    x = Xc0 / Xc2
    y = Xc1 / Xc2

    # distortion (radial k1,k2,k3, tangential p1,p2)
    k1 = dist[0] if dist.size > 0 else 0.0
    k2 = dist[1] if dist.size > 1 else 0.0
    p1 = dist[2] if dist.size > 2 else 0.0
    p2 = dist[3] if dist.size > 3 else 0.0
    k3 = dist[4] if dist.size > 4 else 0.0

    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    radial = 1.0 + k1*r2 + k2*r4 + k3*r6
    x_t = 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
    y_t = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y

    x_d = x*radial + x_t
    y_d = y*radial + y_t

    # intrinsics
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    u = fx * x_d + cx
    v = fy * y_d + cy
    return u, v


@njit(cache=True, fastmath=True, nogil=True)
def _lm_refine_point(
    X_init,
    K_L, dist_L, R_L, t_L,
    K_R, dist_R, R_R, t_R,
    uvL_u, uvR_u,                     # undistorted or raw pixel targets (match your projector)
    max_iters=5, eps=1e-6, lam0=1e-2,
    tol_step=1e-8, tol_improve=1e-6
):
    X = X_init.copy()
    lam = lam0
    J = np.empty((4,3))
    d = np.zeros(3)

    for _ in range(max_iters):
        # residual r = [eLx, eLy, eRx, eRy]
        uL, vL = _project_opencv_numba(K_L, dist_L, R_L, t_L, X)
        uR, vR = _project_opencv_numba(K_R, dist_R, R_R, t_R, X)
        r0 = np.empty(4)
        r0[0] = uL - uvL_u[0]; r0[1] = vL - uvL_u[1]
        r0[2] = uR - uvR_u[0]; r0[3] = vR - uvR_u[1]

        # numeric Jacobian
        for k in range(3):
            d[0]=0.0; d[1]=0.0; d[2]=0.0
            d[k] = eps
            Xp0 = X[0] + d[0]; Xp1 = X[1] + d[1]; Xp2 = X[2] + d[2]
            uL1, vL1 = _project_opencv_numba(K_L, dist_L, R_L, t_L, np.array([Xp0,Xp1,Xp2]))
            uR1, vR1 = _project_opencv_numba(K_R, dist_R, R_R, t_R, np.array([Xp0,Xp1,Xp2]))
            J[0,k] = (uL1 - uL) / eps
            J[1,k] = (vL1 - vL) / eps
            J[2,k] = (uR1 - uR) / eps
            J[3,k] = (vR1 - vR) / eps

        # LM solve: (JᵀJ + λI) δ = -Jᵀ r
        H = J.T @ J
        H[0,0] += lam; H[1,1] += lam; H[2,2] += lam
        g = J.T @ r0
        # solve H δ = -g
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        # small step → stop
        if delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2] < (tol_step*tol_step):
            break

        # candidate
        X_new = np.array([X[0]+delta[0], X[1]+delta[1], X[2]+delta[2]])

        # accept if improves reprojection
        uL2, vL2 = _project_opencv_numba(K_L, dist_L, R_L, t_L, X_new)
        uR2, vR2 = _project_opencv_numba(K_R, dist_R, R_R, t_R, X_new)
        r1_0 = uL2 - uvL_u[0]; r1_1 = vL2 - uvL_u[1]
        r1_2 = uR2 - uvR_u[0]; r1_3 = vR2 - uvR_u[1]
        rms0 = np.sqrt((r0[0]*r0[0]+r0[1]*r0[1]+r0[2]*r0[2]+r0[3]*r0[3]) * 0.25)
        rms1 = np.sqrt((r1_0*r1_0+r1_1*r1_1+r1_2*r1_2+r1_3*r1_3) * 0.25)

        if rms1 < rms0 - tol_improve:
            X = X_new
            lam = max(lam * 0.5, 1e-6)  # reward success
        else:
            lam *= 10.0                 # damp more

    # final RMS
    uLf, vLf = _project_opencv_numba(K_L, dist_L, R_L, t_L, X)
    uRf, vRf = _project_opencv_numba(K_R, dist_R, R_R, X)
    r0 = np.array([uLf-uvL_u[0], vLf-uvL_u[1], uRf-uvR_u[0], vRf-uvR_u[1]])
    rms = float(np.sqrt((r0*r0).mean()))
    return X, rms
