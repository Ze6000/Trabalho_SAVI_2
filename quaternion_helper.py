#########################################################################################################################
#   _____                      _    __          __                  _     _           
#  / ____|                    | |   \ \        / /                 | |   | |          
# | (___  _ __ ___   __ _ _ __| |_   \ \  /\  / /__  __ _ _ __ __ _| |__ | | ___  ___ 
#  \___ \| '_ ` _ \ / _` | '__| __|   \ \/  \/ / _ \/ _` | '__/ _` | '_ \| |/ _ \/ __|
#  ____) | | | | | | (_| | |  | |_     \  /\  /  __/ (_| | | | (_| | |_) | |  __/\__ \
# |_____/|_| |_| |_|\__,_|_|   \__|     \/  \/ \___|\__,_|_|  \__,_|_.__/|_|\___||___/
#                                                                                     
#  Copyright: (c) Copyright 2023
#  License: None. For private and exclusive use of 
#  Wearables Team
#
#  Helper file for quaternion math.
#
#########################################################################################################################

# Base Python Imports
import math

# External Modules Imports
import numpy as np

class QuaternionHelper:
    """ Class with helper functions for quaternion math.

    Quaternions have been first proposed by W. R. Hamilton as an extension of complex numbers.
    A general quaternion is defined as the sum of a scalar q0 and a vector qv = (q1, q2, q3),
        q = q0 + qv = q0 + q1i + q2j + q3k
    and is usually represented as
        q = [q0, q1, q2, q3].
    
    Unit quaternions, that verify the condition
        ||q||2 = (q0^2 + q1^2 + q2^2 + q3^2)^0.5 = 1,
    where ||.||2 is the Euclidean norm, are commonly used to represent rotations in the space (R3).
    The quaternion rotation operator is given by
        q = [q0, q1, q2, q3] = [cos(theta/2), u(sin(theta/2))]
    where u is the axis of rotation and theta is the angle of rotation.

    Unlike the commonly used rotation matrices, that present some redundacy, a quaternion only represents
    one rotation and its geometric meaning is also more obvious and intuitive. 
    
    A vector can also be represented as a quaternion, called a pure quaternion, where q0 = 0 and, thus,
        q = [0, v0, v1, v2].
    
    """


    def __init__(self):
        pass
    

    #####################################################################################################################

    #   ___            _                _              _   _          _             
    #  / _ \ _  _ __ _| |_ ___ _ _ _ _ (_)___ _ _     /_\ | |__ _ ___| |__ _ _ __ _ 
    # | (_) | || / _` |  _/ -_) '_| ' \| / _ \ ' \   / _ \| / _` / -_) '_ \ '_/ _` |
    #  \__\_\\_,_\__,_|\__\___|_| |_||_|_\___/_||_| /_/ \_\_\__, \___|_.__/_| \__,_|
    #                                                       |___/                   

    # Adition
    def compute_quaternion_adition(self, q, p):
        """ Computes a quaternion addition.

        This function computes the adition of two quaternions as,
            q + p = (q0 + p0) + (q1 + p1)i + (q2 + p2)j + (q3 + q3)k

        Arguments:
            q (:obj:`np.array`): a quaternion.
            p (:obj:`np.array`): a quaternion.

        Returns:
            r (:obj:`np.array`): the quaternion resulting from the quaternion addition.
        
        """

        # Check if the quaternions are given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)
        self.check_is_quaternion(p)

        r = q + p
        
        return r
    

    # Multiplication
    def compute_quaternion_multiplication(self, q, p):
        """ Computes a quaternion multiplication.

        This function computes the multiplication of two quaternions as,
            q x p = q0p0 - qv . pv + q0pv + p0qv + q x pv
                  = [q0p0 - q1p1 - q2p2 - q3p3,
                     q0p1 + q1p0 + q2p3 - q3p2,
                     q0p2 - q1p3 + q2p0 + q3p1,
                     q0p3 + q1p2 - q2p1 + q3p0]

        Arguments:
            q (:obj:`np.array`): a quaternion.
            p (:obj:`np.array`): a quaternion.

        Returns:
            r (:obj:`np.array`): the quaternion resulting from the quaternion multiplication.
        
        """

        # Check if the quaternions are given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)
        self.check_is_quaternion(p)

        r    = np.zeros(4)
        r[0] = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3]
        r[1] = q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2]
        r[2] = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1]
        r[3] = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0]
        
        return r


    # Conjugate
    def compute_quaternion_conjugate(self, q):
        """ Computes the quaternion conjugate.

        This function computes the conjugate of a quaternion, as
            q0* = q0 - q = q0 - q1i - q2j - q3k

        An important property of quaternions conjugates is
            (pq)* = q*p*

        Arguments:
            quaternion (:obj:`np.array`): a quaternion.

        Returns:
            conjugate (:obj:`np.array`): the quaternion conjugate.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        conjugate = np.array([q[0], -1 * q[1], -1 * q[2], -1 * q[3]])
        
        return conjugate

    
    # Norm
    def normalize_quaternion(self, q):
        """ Normalizes the given quaternion to unit.

        This function normalizes a quaternion to unit, as
            q = q / |q|
        where the quaternion norm, |q|, is given by
            |q| = (qq*)^0.5 = (q0^2 + q1^2 + q2^2 + q3^2)^0.5 
        
        Some important properties related with the quaternion norm are
            |q|    = |q*|
            |qp|   = |q||p|
            |qp|^2 = |q|^2|p|^2

        Arguments:
            quaternion (:obj:`np.array`): a quaternion.

        Returns:
            quaternion_normalized (:obj:`np.array`): the normalized quaternion.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        if (np.array_equal(q, np.zeros(4))):
            quaternion_normalized = np.zeros(4)
        else:
            norm                  = np.linalg.norm(q)
            quaternion_normalized = q / norm
        
        return quaternion_normalized


    # Vector Norm
    def normalize_vector(self, v):
        """ Normalizes the given vector to unit.

        This function normalizes a vector to unit, as
            v = v / |v|
        where the quaternion norm, |q|, is given by
            |v| = (vx^2 + vy^2 + vz^2)^0.5 

        Arguments:
            v (:obj:`np.array`): a 3D vector (x, y and z components).

        Returns:
            vector_normalized (:obj:`np.array`): the normalized vector.
        
        """

        # Check if the vector is given as expected (np.array, with 1 dimension of length 3)
        self.check_is_vector(v)

        if (np.array_equal(v, np.zeros(3))):
            vector_normalized = np.zeros(3)
        else:
            norm                  = np.linalg.norm(v)
            vector_normalized = v / norm
        
        return vector_normalized


    # Inverse
    def compute_quaternion_inverse(self, q):
        """ Computes the quaternion inverse.

        This function computes the inverse of a quaternion, as
            q^-1 = q* / |q|^2
        
        Some important properties related with the quaternion inverse are
            qq^-1 = q^-1q = 1
            If q is a unit quaternion,
                q^-1 = q*

        Arguments:
            quaternion (:obj:`np.array`): a quaternion.

        Returns:
            inverse (:obj:`np.array`): the quaternion inverse.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        conjugate = self.compute_quaternion_conjugate(q)
        norm      = np.linalg.norm(q)
        inverse   = conjugate / (norm**2)
        
        return inverse


    #####################################################################################################################

    #   ___            _                _            ___     _        _   _             ___                     _           
    #  / _ \ _  _ __ _| |_ ___ _ _ _ _ (_)___ _ _   | _ \___| |_ __ _| |_(_)___ _ _    / _ \ _ __  ___ _ _ __ _| |_ ___ _ _ 
    # | (_) | || / _` |  _/ -_) '_| ' \| / _ \ ' \  |   / _ \  _/ _` |  _| / _ \ ' \  | (_) | '_ \/ -_) '_/ _` |  _/ _ \ '_|
    #  \__\_\\_,_\__,_|\__\___|_| |_||_|_\___/_||_| |_|_\___/\__\__,_|\__|_\___/_||_|  \___/| .__/\___|_| \__,_|\__\___/_|  
    #                                                                                       |_|                             

    # Rotate a vector using quaternions
    def compute_vector_quaternion_rotation(self, v, q):
        """ Rotates a vector by a quaternion.

        This function rotates a vector by a quaternion. This rotation is given by
            qvq* = [v0 * (q0**2 + q1**2 - q2**2 - q3**2) + 2v1 * (q1q2 - q0q3) + 2v2 * (q1q3 + q0q2),
                    v1 * (q0**2 - q1**2 + q2**2 - q3**2) + 2v0 * (q1q2 + q0q3) + 2v2 * (q2q3 - q0q1),
                    v2 * (q0**2 - q1**2 - q2**2 + q3**2) + 2v0 * (q1q3 - q0q2) + 2v1 * (q2q3 + q0q1)]

        Arguments:
            v (:obj:`np.array`): a vector.
            q (:obj:`np.array`): a quaternion.

        Returns:
            u (:obj:`np.array`): the rotated vector.
        
        """
        
        # Check if the vector is given as expected (np.array, with 1 dimension of length 3)
        self.check_is_vector(v)
        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q    = self.normalize_quaternion(q)

        u    = np.zeros(3)
        u[0] = v[0] * (q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2) + 2*v[1] * (q[1]*q[2] - q[0]*q[3]) + 2*v[2] * (q[1]*q[3] + q[0]*q[2])
        u[1] = 2*v[0] * (q[1]*q[2] + q[0]*q[3]) + v[1] * (q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2) + 2*v[2] * (q[2]*q[3] - q[0]*q[1])
        u[2] = 2*v[0] * (q[1]*q[3] - q[0]*q[2]) + 2*v[1] * (q[2]*q[3] + q[0]*q[1]) + v[2] * (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)
        
        return u


    # Quaternion to rotation matrix
    def quaternion_to_rotation_matrix(self, q):
        """ Convertes a quaternion into a rotation matrix.

        A quaternion rotation operator can be represented as a rotation matrix as follows,
            r = [[2q0^2 + 2q1^2 - 1, 2q1q2 - 2q0q3    , 2q1q3 + 2q0q2    ],
                 [2q1q2 + 2q0q3    , 2q0^2 + 2q2^2 - 1, 2q2q3 - 2q0q1    ],
                 [2q1q3 - 2q0q2    , 2q2q3 + 2q0q1    , 2q0^2 + 2q3^2 - 1]]
        
        Note that, for a unit quaternion,
            q0^2 + q1^2 - q2^2 - q3^2 =
            = 2q0^2 + 2q1^2 - 1
            = 1 - 2q2^2 - 2q3^2.

        Arguments:
            q (:obj:`np.array`): a quaternion.
            
        Returns:
            r (:obj:`np.array`): the rotation matrix that represent the quaternion rotation operator.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q  = self.normalize_quaternion(q)
    
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        r = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                      [2*q1*q2 + 2*q0*q3, q0**2 + q2**2 - q1**2 - q3**2, 2*q2*q3 - 2*q0*q1],
                      [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0**2 + q3**2 - q1**2 - q2**2]])
        
        return r
    

    # Quaternion between two vectors
    def compute_quaternion_between_vectors(self, v, w):
        """ Computes the quaternion that represents the rotation between two vectors.

        This function computes the quaternion that represents the rotation between two vectors,
        given by
            q = [cos(theta/2), u*sin(theta/2)],
        where theta = acos(v.w) and u = vxw/||vxw||2, where ||.||2 is the Euclidean norm.
        v and w are normalized vectors and u is the rotation axis between them.
        
        Arguments:
            v (:obj:`np.array`): a vector.
            w (:obj:`np.array`): a vector.
            
        Returns:
            q (:obj:`np.array`): the quaternion that represents the rotation between v and w.
        
        """

        # Check if the vectors are given as expected (np.array, with 1 dimension of length 3)
        self.check_is_vector(v)
        self.check_is_vector(w)

        # Normalize vectors
        v = v / np.linalg.norm(v)
        w = w / np.linalg.norm(w)

        # Compute the angle between the vectors
        angle = math.acos(np.dot(v, w))

        # Compute the axis of rotation
        u_aux = np.cross(v, w)
        u = u_aux / np.linalg.norm(u_aux)

        q = np.array([math.cos(angle / 2), u[0] * math.sin(angle / 2),
                      u[1] * math.sin(angle / 2), u[2] * math.sin(angle / 2)])
        
        return q


    # Skew-symmetric matrix
    def compute_skew_symmetric_matrix(self, q):
        """ Computes the skew-symmetric matrix of a quaternion.

        This function computes the skew-symmetric matrix of a quaternion, given by
            qx = [[  0, -q3,  q2],
                  [ q3,   0, -q1],
                  [-q2,  q1,   0]]
        qx is a skew-symmetric matrix if qx^T = -qx.
        
        Arguments:
            q (:obj:`np.array`): a quaternion.
            
        Returns:
            qx (:obj:`np.array`): the skew-symmetric matrix.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q  = self.normalize_quaternion(q)
        
        qx = np.array([[    0, -q[3],  q[2]],
                       [ q[3],     0, -q[1]],
                       [-q[2],  q[1],     0]])
        
        return qx


    #####################################################################################################################

    #  __  __      _ _   _      _ _         _   _            __       ___  _  __  __                 _   _      _   _           
    # |  \/  |_  _| | |_(_)_ __| (_)__ __ _| |_(_)___ _ _   / _|___  |   \(_)/ _|/ _|___ _ _ ___ _ _| |_(_)__ _| |_(_)___ _ _   
    # | |\/| | || | |  _| | '_ \ | / _/ _` |  _| / _ \ ' \  > _|_ _| | |) | |  _|  _/ -_) '_/ -_) ' \  _| / _` |  _| / _ \ ' \  
    # |_|  |_|\_,_|_|\__|_| .__/_|_\__\__,_|\__|_\___/_||_| \_____|  |___/|_|_| |_| \___|_| \___|_||_\__|_\__,_|\__|_\___/_||_| 
    #                     |_|                                                                                                   

    def compute_multiplication_matrix_left(self, q):
        """ Computes the left multiplication matrix of a quaternion.

        This function computes the left multiplication matrix of a quaternion.
        The multiplication between two quaternions, q x p, can be writen as
            q x p = qLp = pRq
        where qL denotes the left multiplication matrix and qR denotes the right multiplication matrix.
        The left multiplication matrix is defined as:
            qL = | q0    -qv    |
                 | qv q0I3 + qx |
        where qv is the vectorial component of the quaternion, I3 is the 3x3 identity matrix and
        qx the quaternion skew matrix.
        
        This matrix also correspondes to the partial derivative dqp/dp,
            dqp/dp  = |   q0    - qv     |
                      |   qv   q0I3 + qx |
            dqp*/dp = |   q0      qv     |
                      |   qv - q0I3 - qx |
            dqp/dq  = |   q0    - qv     |
                      |   qv   q0I3 - qx |
            dpq*/dp = |   q0      qv     |
                      | - qv   q0I3 + qx |

        Arguments:
            q (:obj:`np.array`): a quaternion.
            
        Returns:
            ql (:obj:`np.array`): the left multiplication matrix.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q  = self.normalize_quaternion(q)

        ql = np.array([[q[0], -q[1], -q[2], -q[3]],
                       [q[1],  q[0], -q[3],  q[2]],
                       [q[2],  q[3],  q[0], -q[1]],
                       [q[3], -q[2],  q[1],  q[0]]])
        
        return ql
    

    def compute_multiplication_matrix_right(self, q):
        """ Computes the right multiplication matrix of a quaternion.

        This function computes the right multiplication matrix of a quaternion.
        The multiplication between two quaternions, q x p, can be writen as
            q x p = qLp = pRq
        where qL denotes the left multiplication matrix and qR denotes the right multiplication matrix.
        The right multiplication matrix is defined as:
            qR = | q0    -qv    |
                 | qv q0I3 - qx |
        where qv is the vectorial component of the quaternion, I3 is the 3x3 identity matrix and
        qx the quaternion skew matrix.
        
        This matrix also correspondes to the partial derivative dqp/dq,
            dqp/dp  = |   q0    - qv     |
                      |   qv   q0I3 + qx |
            dqp*/dp = |   q0      qv     |
                      |   qv - q0I3 - qx |
            dqp/dq  = |   q0    - qv     |
                      |   qv   q0I3 - qx |
            dpq*/dp = |   q0      qv     |
                      | - qv   q0I3 + qx |

        Arguments:
            q (:obj:`np.array`): a quaternion.
            
        Returns:
            qr (:obj:`np.array`): the right multiplication matrix.
        
        """
        
        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q  = self.normalize_quaternion(q)

        qr = np.array([[q[0], -q[1], -q[2], -q[3]],
                       [q[1],  q[0],  q[3], -q[2]],
                       [q[2], -q[3],  q[0],  q[1]],
                       [q[3],  q[2], -q[1],  q[0]]])

        return qr

    
    def compute_quaternion_omega(self, q):
        """ Computes the omega matrix of a quaternion.

        This function computes the omega matrix of a quaternion, defined as:
            qR = |  0   qv |
                 | -qv -qx |
        where qv is the vectorial component of the quaternion and
        qx the quaternion skew matrix.
        
        Arguments:
            q (:obj:`np.array`): a quaternion.
            
        Returns:
            qomega (:obj:`np.array`): the omega matrix.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q  = self.normalize_quaternion(q)

        qomega = np.array([[    0,  q[1],  q[2],  q[3]],
                           [-q[1],     0,  q[3], -q[2]],
                           [-q[2], -q[3],     0,  q[1]],
                           [-q[3],  q[2], -q[1],     0]])

        return qomega


    def compute_quaternion_exponential(self, v):
        """ Computes the quaternion exponential of a vector.

        This function computes the quaternion exponential of a vector, defined as:
            q = expq(v) = [cos(||v||2), sin(||v||2)(v/||v||2)]
        where expq(v) represents the quaternion exponential of the vector and ||.||2 is the Euclidean norm.

        Arguments:
            v (:obj:`np.array`): a vector.
            
        Returns:
            q (:obj:`np.array`): the quaternion representing the quaternion exponential.
        
        """

        # Check if the vector is given as expected (np.array, with 1 dimension of length 3)
        self.check_is_vector(v)

        norm = np.linalg.norm(v)
        
        q = np.array([            np.cos(norm),
                      v[0]/norm * np.sin(norm),
                      v[1]/norm * np.sin(norm),
                      v[2]/norm * np.sin(norm)])
        
        return q


    def compute_derivative_rotation_matrix_quaternion(self, q):
        """ Computes the derivative of a quaternion based rotation matrix with respect to the quaternion.
        
        This function computes the derivative of a quaternion based rotation matrix (nbR)
        with respect to the quaternion (nbq):
            dR = dnbR / dnbq.
        Note that this results in a 3D matrix of shape 4x3x3, since dRi = dnbR(q) / dnbqi, i = {0, 1, 2, 3}.
        For example,
            dR0 = dnbR(q) / dnbq0 = 2 * | q0 -q3  q2 |
                                        | q3  q0 -q1 |
                                        |-q2  q1  q0 |

        Arguments:
            q (:obj:`np.array`): a quaternion.
            
        Returns:
            dR (:obj:`np.array`): the derivative of the rotation matrix with respect to the quaternion.
        
        """

        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q  = self.normalize_quaternion(q)
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        dR = 2 * np.array([[[ q0, -q3,  q2], [ q3,  q0, -q1], [-q2,  q1,  q0]],  # q0
                           [[ q1,  q2,  q3], [ q2, -q1, -q0], [ q3,  q0, -q1]],  # q1
                           [[-q2,  q1,  q0], [ q1,  q2,  q3], [-q0,  q3, -q2]],  # q2
                           [[-q3, -q0,  q1], [ q0, -q3,  q2], [ q1,  q2,  q3]]]) # q3
        
        return dR


    def compute_multiplication_derivative_by_vector(self, v, q):
        """ Computes the multiplication between the derivative of a rotation matrix and a vector.
        
        This function computes the multiplication of the derivative of a quaternion based rotation matrix
        by a vector, as
            dRv = |dR[0].v, dR[1].v, dR[2].v, dR[3].v|,
        where dR = dbnR / dbnq, thus resulting in a 3x4 matrix.
        
        Arguments:
            q (:obj:`np.array`): a quaternion.
            v (:obj:`np.array`): a vector.
            
        Returns:
            dRv (:obj:`np.array`): the multiplication between the derivative of the matrix and the vector.
        
        """

        # Check if the vector is given as expected (np.array, with 1 dimension of length 3)
        self.check_is_vector(v)
        # Check if the quaternion is given as expected (np.array, with 1 dimension of length 4)
        self.check_is_quaternion(q)

        # Normalize quaternion to unit
        q   = self.normalize_quaternion(q)
    
        dR  = self.compute_derivative_rotation_matrix_quaternion(q)
        dRv = np.zeros([3, 4])

        dRv[:, 0] = np.dot(dR[0], v)
        dRv[:, 1] = np.dot(dR[1], v)
        dRv[:, 2] = np.dot(dR[2], v)
        dRv[:, 3] = np.dot(dR[3], v)
        
        return dRv

    #####################################################################################################################

    def check_array_type(self, array):
        """ Checks the type of an array and its elements.

        This function checks if an array and all its elements are of the type np.int32 or np.float64.
    
        Arguments:
            array (:obj:`np.array`): an array.
        
        Raises:
            TypeError: if the array or some element of the array is not a np.int32 or a np.float64.
        
        """

        if (type(array) is not np.ndarray):
            raise TypeError('Error 1 | The array ' + str(array) + ' is not a np.ndarray: ' + str(type(array)))

        for i in range (0, len(array)):
            #print(len(array))
            if (type(array[i]) is not np.int32) and (type(array[i]) is not np.float64):
                raise TypeError('Error 2 | The element ' + str(i) + ' of the array is not a np.int32 or np.float64: ' +
                                str(array[i]) + ' (type ' + str(type(array[i])) + ')')
                                
    
    def check_number_dimensions_and_shape(self, array, number_dimensions, length_dimension, dimension = 0):
        """ Checks the number of dimensions and the shape of an array.

        This function checks the number of dimensions of an array and the length of a given dimension.
    
        Arguments:
            array (:obj:`np.array`): an array.
            number_dimensions (int): the number of dimensions expected for the array.
            length_dimension (int): the length of an expected dimension.
            dimension (int): the dimension whose length will be checked (predefined: 0).
        
        Raises:
            ValueError: if the number of dimensions and/or the length of the dimensions does not correspond
                        to the expected.
        
        """

        n_dim = array.ndim
        shape = array.shape

        if number_dimensions <= dimension:
            raise ValueError('Error 1 | The dimension to extract the length (' + str(dimension) +
                             ') must be smaller than the number of dimensions (' + str(n_dim) + ')')

        if n_dim is not number_dimensions:
            raise ValueError('Error 2 | The array does not present the expected number of dimensions: ' + str(n_dim) +
                             '; Expected: ' + str(number_dimensions))

        if shape[dimension] is not length_dimension:
            raise ValueError('Error 3 | The dimension ' + str(dimension) + ' does not present the expected length: '+
                             str(shape[dimension]) + '; Expected: ' + str(length_dimension))


    def check_is_vector(self, vector):
        """ Checks if an array is a vector.

        This function checks if an array present only 1 dimension of length 3 and
        if all elements are of the type int or float.
    
        Arguments:
            vector (:obj:`np.array`): a vector.
        
        Raises:
            ValueError: if the number of dimensions and/or the length of the dimensions does not correspond
                        to the expected.
            TypeError: if some element of the array is not a int or a float.
        
        """

        self.check_array_type(vector)
        self.check_number_dimensions_and_shape(vector, 1, 3)


    def check_is_quaternion(self, quaternion):
        """ Checks if an array is a quaternion.

        This function checks if an array present only 1 dimension of length 4 and
        if all elements are of the type int or float.
    
        Arguments:
            quaternion (:obj:`np.array`): a quaternion.
        
        Raises:
            ValueError: if the number of dimensions and/or the length of the dimensions does not correspond
                        to the expected.
            TypeError: if some element of the array is not a int or a float.
        
        """

        self.check_array_type(quaternion)
        self.check_number_dimensions_and_shape(quaternion, 1, 4)