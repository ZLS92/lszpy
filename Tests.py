#!/usr/bin/env python3
# ==============================================================================

# =============================================================================
# In[0]: 
# File Headers
cell_sep = "\n# =============================================================================\n"
print( cell_sep + "Cell 0: File header" )

file_headers ="""
\tCreated on Mon May 16 17:57:00 2025

\t@author: Zampa Luigi Sante
\t@email_1: lzampa@ogs.it
\t@org: National Institute of Oceanography and Applied Geophysics - OGS
"""
print( file_headers )

# =============================================================================
# In[1]: 

# Import base modules
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import importlib
import grav_model as gm 

# ==============================================================================
# In[5]:
# Compute Topo Effect
# print( cell_sep + "Cell 5: Compute Topo Effect\n")

# # Coordinate del prisma in formato [west, east, south, north, bottom, top]
# prism = np.array([[0, 1000, 0, 1000, 0, 1000]])  # un cubo 1x1x1 m
prism = np.array([[0, 20, 0, 10, -10, 0]])
prism = np.array([[-50, 50, -50, 50, 0, 50]])

# prism = np.array([[0, 50, 0, 50, 0, 50]])


# # Punto di osservazione (sopra il centro del prisma)
obs_point = np.array([[0.0, 0.0, 70.0]])
# # obs_point = np.array([[0.5, 0.5, 2.0]]) 

# # Densità in kg/m^3
# density = 2670

nagy_model = gm.nagy(coordinates=obs_point, prism=prism, density=2670)
gz_nagy = nagy_model.gz()
print( gz_nagy )

Gc = gm.G

def prism_to_polyhedra(prisms):
    """
    Converti uno o più prismi in una lista di poliedri (corners + faces),
    verificando le normali outward per ogni faccia.

    Parametri
    ----------
    prisms : ndarray (N, 6)
        Ogni riga: [xmin, xmax, ymin, ymax, zmin, zmax]

    Ritorna
    -------
    polyhedra : list of dict
        Ogni elemento: {'corners': ndarray, 'faces': ndarray}
    """
    polyhedra = []

    for prism in prisms:
        xmin, xmax, ymin, ymax, zmin, zmax = prism

        corners = np.array([
            [xmin, ymin, zmax],  # 0: top front left
            [xmax, ymin, zmax],  # 1: top front right
            [xmax, ymax, zmax],  # 2: top back right
            [xmin, ymax, zmax],  # 3: top back left
            [xmin, ymin, zmin],  # 4: bottom front left
            [xmax, ymin, zmin],  # 5: bottom front right
            [xmax, ymax, zmin],  # 6: bottom back right
            [xmin, ymax, zmin],  # 7: bottom back left
        ], dtype=float ) 

        faces = [
            [4, 0, 1, 2, 3],  # Top
            [4, 4, 5, 6, 7],  # Bottom
            [4, 0, 4, 5, 1],  # Front
            [4, 1, 5, 6, 2],  # Right
            [4, 2, 6, 7, 3],  # Back
            [4, 3, 7, 4, 0],  # Left
        ]

        corrected_faces = []
        poly_center = np.mean(corners, axis=0)

        for face in faces:
            nverts = face[0]
            idx = np.array(face[1:])
            verts = corners[idx]

            v1 = verts[1] - verts[0]
            v2 = verts[2] - verts[0]
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm != 0:
                normal /= normal_norm

            centroid = np.mean(verts, axis=0)
            to_center = centroid - poly_center

            # Se la normale punta verso l'interno: inverti ordine vertici
            if np.dot(normal, to_center) < 0:
                idx = idx[::-1]

            corrected_faces.append([nverts] + list(idx))

        polyhedra.append({
            'corners': corners,
            'faces': np.array(corrected_faces)
        })

    return polyhedra

polyhedra = prism_to_polyhedra( prism )

def anglegravi_exact(p1, p2, p3, Un):
    """
    Compute the angle between planes (O-p1-p2) and (O-p2-p3),
    used to compute the solid angle seen from the origin O.

    Parameters
    ----------
    p1, p2, p3 : array-like, shape (3,)
        Points on the polygon in CCW order.
    Un : array-like, shape (3,)
        Unit outward normal vector of the face.

    Returns
    -------
    ang : float
        Angle in radians.
    perp : int
        Sign of the perpendicular projection (1 or -1), 0 if undefined.
    """

    inout = np.sign(np.sum(Un * p1))

    x2, y2, z2 = p2

    if inout == 0:
        ang = 0.0
        perp = 1
    else:
        if inout > 0:  # seen from inside; interchange p1 and p3
            x1, y1, z1 = p3
            x3, y3, z3 = p1
        else:          # seen from outside; keep p1 and p3
            x1, y1, z1 = p1
            x3, y3, z3 = p3

        # Compute normals
        n1 = np.array([
            y2 * z1 - y1 * z2,
            x1 * z2 - x2 * z1,
            x2 * y1 - x1 * y2
        ])
        n2 = -np.array([
            y3 * z2 - y2 * z3,
            x2 * z3 - x3 * z2,
            x3 * y2 - x2 * y3
        ])

        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        perp = np.sign(np.dot([x3, y3, z3], n1))
        r = np.dot(n1, n2)
        r = np.clip(r, -1.0, 1.0)  # safety against rounding errors
        ang = np.arccos(r)

        if perp < 0:
            ang = 2 * np.pi - ang

    return ang, perp

def gravicalc2(X, Y, Z, polyhedra, dens):
    """
    Calcola il campo gravitazionale Gx, Gy, Gz in un punto X,Y,Z
    sommando l'effetto di tutti i poliedri presenti in `polyhedra`.

    Parametri
    ----------
    X, Y, Z : ndarray, shape (npro, nstn)
        Coordinate dei punti di osservazione.
    polyhedra : list of dict
        Ogni dict ha 'corners' e 'faces'.
    dens : float
        Densità.

    Returns
    -------
    Gx, Gy, Gz : ndarray
    """
    npro, nstn = X.shape
    Gx, Gy, Gz = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)

    for poly in polyhedra:
        corners = poly['corners']
        faces = poly['faces']

        Ncor = corners.shape[0]
        Nf = faces.shape[0]

        # --------------------------
        # Costruisci Edge e Normali
        Nedges = np.sum(faces[:, 0])
        Edge = np.zeros((Nedges, 8))

        edge_counter = 0
        for f in range(Nf):
            nverts = faces[f, 0]
            verts = faces[f, 1:1 + nverts]
            indx = np.append(verts, verts[0])
            for t in range(nverts):
                edgeno = edge_counter
                edge_counter += 1
                ends = [indx[t], indx[t+1]]
                p1 = corners[ends[0]]
                p2 = corners[ends[1]]
                V = p2 - p1
                L = np.linalg.norm(V)
                Edge[edgeno, 0:3] = V
                Edge[edgeno, 3] = L
                Edge[edgeno, 6:8] = ends

        Un = np.zeros((Nf, 3))
        for t in range(Nf):
            ss = np.zeros(3)
            nverts = faces[t, 0]
            face_verts = faces[t, 1:1 + nverts]
            for t1 in range(1, nverts - 1):
                v1 = corners[face_verts[t1 + 1]] - corners[face_verts[0]]
                v2 = corners[face_verts[t1]] - corners[face_verts[0]]
                ss += np.cross(v2, v1)
            norm_ss = np.linalg.norm(ss)
            Un[t, :] = ss / norm_ss if norm_ss != 0 else ss

        # --------------------------
        # Loop punti di osservazione
        cor = np.zeros((Ncor, 3))
        for pr in range(npro):
            for st in range(nstn):
                opt = np.array([X[pr, st], Y[pr, st], Z[pr, st]])
                fsign = np.zeros(Nf)
                Omega = np.zeros(Nf)
                for t in range(Ncor):
                    cor[t, :] = corners[t, :] - opt

                for f in range(Nf):
                    nsides = faces[f, 0]
                    cors = faces[f, 1:1 + nsides]
                    Edge[:, 4:6] = 0
                    indx = list(range(nsides)) + [0, 1]
                    crs = np.array([cor[cors[t], :] for t in range(nsides)])
                    fsign[f] = np.sign(np.dot(Un[f, :], crs[0, :]))
                    dp1 = np.dot(crs[indx[0]], Un[f, :])
                    dp = abs(dp1)

                    if dp == 0:
                        Omega[f] = 0
                    else:
                        W = 0
                        for t_angle in range(nsides):
                            p1 = crs[indx[t_angle]]
                            p2 = crs[indx[t_angle + 1]]
                            p3 = crs[indx[t_angle + 2]]
                            ang, _ = anglegravi_exact(p1, p2, p3, Un[f, :])
                            W += ang
                        W -= (nsides - 2) * np.pi
                        Omega[f] = -fsign[f] * W

                    PQR = np.zeros(3)
                    for t in range(nsides):
                        p1 = crs[indx[t]]
                        p2 = crs[indx[t + 1]]
                        Eno = sum(faces[:f, 0]) + t
                        if Edge[Eno, 5] == 1:
                            I = Edge[Eno, 4]
                            V = Edge[Eno, 0:3]
                            PQR += I * V
                        else:
                            chsgn = 1
                            if np.allclose(p1 / np.linalg.norm(p1), p2 / np.linalg.norm(p2)):
                                if np.linalg.norm(p1) > np.linalg.norm(p2):
                                    chsgn = -1
                                    p1, p2 = p2, p1
                            V = Edge[Eno, 0:3]
                            L = Edge[Eno, 3]
                            L2 = L * L
                            b = 2 * np.dot(V, p1)
                            r1 = np.linalg.norm(p1)
                            r12 = r1 * r1
                            b2 = b / (2 * L)
                            if r1 + b2 == 0:
                                V = -V
                                b = 2 * np.dot(V, p1)
                                b2 = b / (2 * L)
                            if r1 + b2 != 0:
                                I = (1 / L) * np.log((np.sqrt(L2 + b + r12) + L + b2) / (r1 + b2))
                                s = np.where((Edge[:, 6] == Edge[Eno, 7]) & (Edge[:, 7] == Edge[Eno, 6]))[0]
                                I *= chsgn
                                Edge[Eno, 4] = I
                                Edge[Eno, 5] = 1
                                if s.size > 0:
                                    Edge[s[0], 4] = I
                                    Edge[s[0], 5] = 1
                                PQR += I * V

                    l, m, n = Un[f, :]
                    p, q, r = PQR
                    if dp != 0:
                        Gx[pr, st] += -dens * Gc * dp1 * (l * Omega[f] + n * q - m * r)
                        Gy[pr, st] += -dens * Gc * dp1 * (m * Omega[f] + l * r - n * p)
                        Gz[pr, st] += +dens * Gc * dp1 * (n * Omega[f] + m * p - l * q)

    Gx, Gy, Gz = Gx * 1e5, Gy * 1e5, Gz * 1e5

    return Gx, Gy, Gz 



X = np.array( [ [ obs_point[0][0] ] ] )
Y = np.array( [ [ obs_point[0][1] ] ] )
Z = np.array( [ [ obs_point[0][2] ] ] )
gz_poly = gravicalc2(X, Y, Z, polyhedra, 2670)
print(gz_poly)
