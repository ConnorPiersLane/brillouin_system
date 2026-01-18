import numpy as np

def _unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length direction")
    return v / n

def build_world_from_points(
    O,                 # origin point (3,)
    Xp=None, Yp=None, Zp=None,  # optional points defining axis directions from O
    prefer="xyz",      # priority for which axes to trust most
    right_handed=True  # enforce right-handed basis
):
    """
    Returns (R_world_rig, T_world_rig) where:
      p_world = R_world_rig @ p_rig + T_world_rig

    O, Xp, Yp, Zp are given in the *current rig frame* (e.g., left-cam frame).
    Provide O, plus at least TWO axis-defining points among Xp/Yp/Zp.

    Examples:
      - Give O, Xp, Yp  (Z is derived as X×Y)
      - Give O, Xp, Zp  (Y is derived as Z×X)
      - Give O, Yp, Zp  (X is derived as Y×Z)
      - If all three are provided, we orthonormalize following `prefer`.
    """
    O = np.asarray(O, float).reshape(3)

    have = {"x": Xp is not None, "y": Yp is not None, "z": Zp is not None}
    if sum(have.values()) < 2:
        raise ValueError("Provide at least two of Xp, Yp, Zp")

    X = _unit(np.asarray(Xp, float) - O) if Xp is not None else None
    Y = _unit(np.asarray(Yp, float) - O) if Yp is not None else None
    Z = _unit(np.asarray(Zp, float) - O) if Zp is not None else None

    # Helper: orthonormalize with a preferred order
    def orthonormalize(X, Y, Z, prefer):
        if prefer == "xyz":
            if X is None:
                # derive X from Y,Z
                X = _unit(np.cross(Y, Z))
            if Y is None and Z is not None:
                Y = _unit(np.cross(Z, X))
            if Z is None and Y is not None:
                Z = _unit(np.cross(X, Y))
            # Gram-Schmidt refine
            X = _unit(X)
            Y = _unit(Y - X * (X @ Y)) if Y is not None else _unit(np.cross(Z, X))
            Z = _unit(np.cross(X, Y))
        elif prefer == "xzy":
            if X is None:
                X = _unit(np.cross(Y, Z))
            if Z is None and Y is not None:
                Z = _unit(np.cross(X, Y))
            if Y is None and Z is not None:
                Y = _unit(np.cross(Z, X))
            X = _unit(X)
            Z = _unit(Z - X * (X @ Z)) if Z is not None else _unit(np.cross(X, Y))
            Y = _unit(np.cross(Z, X))
        elif prefer == "yzx":
            if Y is None:
                Y = _unit(np.cross(Z, X))
            if Z is None and X is not None:
                Z = _unit(np.cross(X, Y))
            if X is None and Z is not None:
                X = _unit(np.cross(Y, Z))
            Y = _unit(Y)
            Z = _unit(Z - Y * (Y @ Z)) if Z is not None else _unit(np.cross(X, Y))
            X = _unit(np.cross(Y, Z))
        else:
            raise ValueError("Unsupported prefer order")
        return X, Y, Z

    X, Y, Z = orthonormalize(X, Y, Z, prefer=prefer)

    # Enforce right-handedness if requested
    if right_handed:
        if np.dot(np.cross(X, Y), Z) < 0:
            # Flip the least trusted axis (here we flip Z)
            Z = -Z

    # Build rotation (columns are basis vectors expressed in rig frame)
    R_world_rig = np.column_stack((X, Y, Z))  # 3x3
    # Translation to move origin from rig to world: world origin is at O
    # Mapping: p_world = R * p_rig + T. We want p_world(O) = 0  =>  0 = R*O + T
    T_world_rig = -R_world_rig @ O

    return R_world_rig, T_world_rig
