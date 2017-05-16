import numpy as np

header = "%+6s\t%+8s\t%+8s\t%+8s\t%+8s\t%+8s" % ("#", "f(x)", "g(x)", "F(x)", "gradf(x)", "nonzeros")
contentFormat = "%6d\t%4e\t%4e\t%4e\t%4e\t%6d"
printHeader = lambda n_iter: (n_iter == 1)
printContent = lambda n_iter: (n_iter < 5) or ((n_iter > 3) and (n_iter % 10 == 0))

def fista(input_size, eval_fun, regulariser,
          regulariser_function=None, thresholding_function=None, initial_x=0,
          L0=1., eta=2., update_L=True,
          verbose=1, verbose_output=0):
    """
    FISTA (Fast Iterative Shrinkage Thresholding Algorithm) is an algorithm to solve the convex minimization of
    y = f(x) + regulariser * g(x) with g(x) can be a continuous and non-differentiable function, such as L1 norm or total variation in compressed sensing.
    If f(x) = || Ax - b ||^2, then the gradient is Df(x) = 2 * A.T * (Ax - b).
    This is from A. Beck and M. Teboulle's paper in 2009: A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
    
    Arguments
    ---------
        input_size: (int or tuple of ints) shape of the signal
        eval_fun: (function with two outputs) evaluation function to calculate f(x) and its gradient, Df(x)
        regulariser: (float) regulariser weights to be multiplied with the regulariser function, g(x)
        regulariser_function: (function or string) the regulariser function, g(x), or string to specify the regulariser, such as "l1" or "tv" (default: reg_l1)
        thresholding_function: (function) function to apply thresholding (or denoising) to the signal in the gradient descent.
            This is ignored if regulariser function is a string (default: soft_threshold_l1)
        initial_x: (int or array) 0 for zeros, 1 for random, or array with shape = input_size to specify the initial guess of the signal (default: 0)
        L0: (float) initial guess of the inverse of step size (default: 1)
        eta: (float) the increment of L0 if the step size is too large (default: 2)
        update_L: (bool or int) flag whether to update L or keep it fix (default: True)
        verbose: (bool or int) flag to show the iteration update (default: True)
        verbose_output: (bool or int) indicate whether the function should return the full information or just the signal (default: False)
    
    Returns
    -------
        The signal if (verbose_output == False) or a dictionary with the output signal (x), number of iterations (n_iter), 
            evaluation function (fx), gradient (gradx), and regulariser function (gx) values
    """
    ############################### argument check ###############################
    eta = float(eta)
    initial_x = _get_initial_x(initial_x, input_size)
    regulariser_fun, thresholding_fun = _get_regulariser(regulariser, regulariser_function, thresholding_function)
    
    ############################### initialisation ###############################
    L = float(L0)
    x = initial_x
    y_next = x
    t_next = 1.
    F_prev = None
    
    ############################### main iteration ###############################
    n_iter = 1
    while True:
        # obtain the parameters from the previous iteration
        L_prev = L
        x_prev = x
        y = y_next
        t = t_next
        
        # calculate the function and the gradient of the evaluation function
        f_y, grad_y = eval_fun(y)
        g_y = regulariser_fun(y)
        F = f_y + g_y
        
        # print the message
        if verbose == 1:
            if printHeader(n_iter): print(header)
            if printContent(n_iter): print(contentFormat % (n_iter, f_y, g_y, F, np.sum(np.abs(grad_y)), np.sum(y > 0)))
        
        # check convergence and update F_prev
        if F_prev != None and np.abs(F - F_prev) / (1e-10+np.abs(F_prev)) < 1e-6: break
        F_prev = F
        
        # find i_k for L=eta**i_k*L such that F(pL(yk)) <= QL(pL(yk), yk)
        L_test = L_prev
        while True:
            pLy = thresholding_fun(y - 1./L_test * grad_y, float(regulariser)/L_test) # gradient descent with thresholding afterwards
            if not update_L: break
            
            pLy_min_y = pLy - y
            reg_pLy = regulariser_fun(pLy)
            
            f_pLy, grad_pLy = eval_fun(pLy)
            F_pLy = f_pLy + reg_pLy
            Q_pLy = f_y + np.sum(pLy_min_y * grad_y) + L_test/2. * np.sum(pLy_min_y * pLy_min_y) + reg_pLy
            
            if (F_pLy <= Q_pLy): break
            L_test *= eta
        
        # calculate the next parameters
        L = L_test
        x = pLy
        t_next = (1. + np.sqrt(1 + 4.*t**2))/2.
        y_next = x + ((t - 1.) / t_next) * (x - x_prev)
        n_iter += 1
    
    ############################### output ###############################
    if verbose_output:
        return {"x": y, "n_iter": n_iter, "fx": f_y, "gradx": grad_y, "gx": g_y}
    else:
        return y

def twist(input_size, eval_fun, regulariser,
          regulariser_function=None, thresholding_function=None, initial_x=0,
          alpha=0, beta=0, lmbda1=1e-4, max_eigval=2., monotone=True,
          verbose=1, verbose_output=0):
    """
    TwIST (Two Steps Iterative Shrinkage Thresholding) is an algorithm to solve the convex minimization of
    y = f(x) + regulariser * g(x) with g(x) can be a continuous and non-differentiable function, such as L1 norm or total variation in compressed sensing.
    If f(x) = || Ax - b ||^2, then the gradient is Df(x) = 2 * A.T * (Ax - b).
    This is from J. M. Bioucas-Dias and M. A. T. Figueiredo's paper in 2007: A New TwIST: Two-Step Iterative Shrinkage/Thresholding Algorithms for Image Restoration.
    
    Arguments
    ---------
        input_size: (int or tuple of ints) shape of the signal
        eval_fun: (function with two outputs) evaluation function to calculate f(x) and its gradient, Df(x)
        regulariser: (float) regulariser weights to be multiplied with the regulariser function, g(x)
        regulariser_function: (function or string) the regulariser function, g(x), or string to specify the regulariser, such as "l1" or "tv" (default: reg_l1)
        thresholding_function: (function) function to apply thresholding (or denoising) to the signal in the gradient descent.
            This is ignored if regulariser function is a string (default: soft_threshold_l1)
        initial_x: (int or array) 0 for zeros, 1 for random, or array with shape = input_size to specify the initial guess of the signal (default: 0)
        alpha: (float between 0 and 1) a step size in the algorithm (see eq. 16) (default: 2. / (1. + np.sqrt(1. - rho0^2)))
        beta: (float between 0 and 1) a step size in the algorithm (see eq. 16) (default: alpha * 2. / (lmbda1 + 1))
        lmbda1: (float) chi parameter in the algorithm (see eq. 20).
            Set lmbda1 = 1e-4 for severely ill conditioned problem, 1e-2 for mildly ill, and 1 for unitary operator (default: 1e-4)
        max_eigval: (float) the guessed largest eigenvalue of A.T*T which equals to the inverse of the step size (default: 2)
        monotone: (bool or int) indicate whether to enforce monotonic function's value decrease in every iteration (default: True)
        verbose: (bool or int) flag to show the iteration update (default: True)
        verbose_output: (bool or int) indicate whether the function should return the full information or just the signal (default: False)
    
    Returns
    -------
        The signal if (verbose_output == False) or a dictionary with the output signal (x), number of iterations (n_iter), 
            evaluation function (fx), gradient (gradx), and regulariser function (gx) values
    """
    
    ############################### argument check ###############################
    # twist parameters
    lmbdaN = 1.
    rho0 = (1. - lmbda1/lmbdaN) / (1. + lmbda1/lmbdaN)
    if alpha == 0:
        alpha = 2. / (1. + np.sqrt(1. - rho0*rho0))
    if beta == 0:
        beta = alpha * 2. / (lmbda1 + lmbdaN)
    initial_x = _get_initial_x(initial_x, input_size)
    regulariser_fun, thresholding_fun = _get_regulariser(regulariser, regulariser_function, thresholding_function)
    
    ############################### initialisation ###############################
    regulariser = float(regulariser)
    x = initial_x
    x_mid = initial_x
    
    # calculate the initial objective function
    y, grad_y = eval_fun(x)
    F_prev = y + regulariser_fun(x)
    
    ############################### main iteration ###############################
    n_iter = 1
    twist_iter = 0
    while True:
        while True:
            y_mid, grad_y_mid = eval_fun(x_mid)
            thresholding_x_mid = thresholding_fun(x_mid - grad_y_mid/max_eigval, regulariser/max_eigval)
            
            if twist_iter == 0: # do an IST iteration
                y_next, grad_y_next = eval_fun(thresholding_x_mid)
                F = y_next + regulariser_fun(thresholding_x_mid)
                
                # if not decreasing, then increase the max_eigval by 2
                if F > F_prev:
                    max_eigval *= 2.
                    if max_eigval > 1e10: break
                    # print(max_eigval)
                else:
                    twist_iter = 1
                    x = x_mid
                    x_mid = thresholding_x_mid
                    break
            else:
                # perform TwIST
                z = (1 - alpha) * x + (alpha - beta) * x_mid + beta * thresholding_x_mid
                y_next, grad_y_next = eval_fun(z)
                F = y_next + regulariser_fun(z)
                
                # if F > F_prev and enforcing monotone, do an IST iteration with double eigenvalue
                if (F > F_prev) and monotone:
                    twist_iter = 0
                else:
                    x = x_mid
                    x_mid = z
                    break
        
        if max_eigval > 1e10: break
        
        # print the message
        if verbose == 1:
            if printHeader(n_iter): print(header)
            if printContent(n_iter): print(contentFormat % (n_iter, y_next, F-y_next, F, np.sum(np.abs(grad_y_next)), np.sum(x_mid > 0)))
        
        # check convergence and update F_prev
        if F_prev != None and np.abs(F - F_prev) / (1e-10+np.abs(F_prev)) < 1e-6: break
        F_prev = F
        
        # prepare the variables for the next iteration
        n_iter += 1
    
    ############################### output ###############################
    if verbose_output:
        return {"x": x_mid, "n_iter": n_iter, "fx": y_next, "gradx": grad_y_next, "gx": F-y_next}
    else:
        return x_mid

def owlqn(input_size, eval_fun, regulariser,
          initial_x=0, m=10, beta=0.7, gamma=0.8,
          verbose=1, verbose_output=0):
    """
    OWL-QN (Orthant Wise Limited-memory Quasi Newton) is an algorithm to solve the convex minimization of y = f(x) + regulariser * |x|_1.
    This algorithm is based on L-BFGS that takes the second order (estimated Hessian matrix) during the search.
    If f(x) = || Ax - b ||^2, then the gradient is Df(x) = 2 * A.T * (Ax - b).
    This is from G. Andrew and J. Gao's paper in 2007: Scalable training of L1-regularized log-linear models.
    
    Arguments
    ---------
        input_size: (int or tuple of ints) shape of the signal
        eval_fun: (function with two outputs) evaluation function to calculate f(x) and its gradient, Df(x)
        regulariser: (float) regulariser weights to be multiplied with the regulariser function, g(x)
        regulariser_function: (function or string) the regulariser function, g(x), or string to specify the regulariser, such as "l1" or "tv" (default: reg_l1)
        thresholding_function: (function) function to apply thresholding (or denoising) to the signal in the gradient descent.
            This is ignored if regulariser function is a string (default: soft_threshold_l1)
        initial_x: (int or array) 0 for zeros, 1 for random, or array with shape = input_size to specify the initial guess of the signal (default: 0)
        alpha: (float between 0 and 1) a step size in the algorithm (see eq. 16) (default: 2. / (1. + np.sqrt(1. - rho0^2)))
        beta: (float between 0 and 1) a step size in the algorithm (see eq. 16) (default: alpha * 2. / (lmbda1 + 1))
        lmbda1: (float) chi parameter in the algorithm (see eq. 20).
            Set lmbda1 = 1e-4 for severely ill conditioned problem, 1e-2 for mildly ill, and 1 for unitary operator (default: 1e-4)
        max_eigval: (float) the guessed largest eigenvalue of A.T*T which equals to the inverse of the step size (default: 2)
        monotone: (bool or int) indicate whether to enforce monotonic function's value decrease in every iteration (default: True)
        verbose: (bool or int) flag to show the iteration update (default: True)
        verbose_output: (bool or int) indicate whether the function should return the full information or just the signal (default: False)
    
    Returns
    -------
        The signal if (verbose_output == False) or a dictionary with the output signal (x), number of iterations (n_iter), 
            evaluation function (fx), gradient (gradx), and regulariser function (gx) values
    """
    ############################### argument check ###############################
    initial_x = _get_initial_x(initial_x, input_size)
    
    _regulariser_fun = lambda x: regulariser * reg_l1(np.abs(x))
    _constraint_orthant = lambda var, orthant: var * (np.sign(var) == orthant)
    
    ############################### initialisation ###############################
    S = [] # list to store the displacements
    Y = [] # list to store the differences in gradient
    x_next = initial_x
    f_x_next, grad_x_next = eval_fun(x_next)
    g_x_next = _regulariser_fun(x_next)
    alpha = 1.
    
    def _compute_direction_lbfgs(S, Y, grad):
        # S: displacement history
        # Y: difference in gradient history
        # grad: first order gradient direction
        # compute the gradient for maximisation
        
        N = len(Y)
        if N == 0: return grad
        
        q = grad
        alphas = [None for i in range(len(Y))]
        rhos = [None for i in range(len(Y))]
        for i in range(len(Y)-1,-1,-1):
            # compute the rho, alpha (and save them), and update q
            rhos[i] = 1./np.sum(Y[i] * S[i])
            alphas[i] = rhos[i] * np.sum(S[i] * q)
            q = q - alphas[i] * Y[i]
        
        H_0 = 1. * np.sum(S[-1] * Y[-1]) / np.sum(Y[-1] * Y[-1])
        z = H_0 * q
        
        for i in range(len(Y)):
            beta = rhos[i] * np.sum(Y[i] * z)
            z = z + S[i] * (alphas[i] - beta)
        
        return z
    
    ############################### main iteration ###############################
    n_iter = 1
    while True:
        f_x = f_x_next
        grad_x = grad_x_next
        g_x = g_x_next
        F_x = f_x + g_x
        x = x_next
        
        # print the message
        if verbose == 1:
            if printHeader(n_iter): print(header)
            if printContent(n_iter): print(contentFormat % (n_iter, f_x, g_x, F_x, np.sum(np.abs(grad_x)), np.sum(x > 0)))
        
        # compute the pseudo-gradient of f by computing the directional gradient of f first
        sgnx = np.sign(x)
        sgnx_0 = (sgnx == 0)
        grad_pos_x = grad_x + regulariser * (sgnx + sgnx_0)
        grad_neg_x = grad_x + regulariser * (sgnx - sgnx_0)
        pgrad_x = grad_pos_x * (grad_pos_x < 0) + grad_neg_x * (grad_neg_x > 0)
        
        # choose the orthant
        orthant = np.sign(x) * (x != 0) - np.sign(pgrad_x) * (x == 0)
        
        # compute the inverse hessian multiplied by the pseudo gradient from the S & Y history
        grad_2nd = _compute_direction_lbfgs(S, Y, pgrad_x)
        
        # constraint the gradient direction
        grad_2nd_orthant = _constraint_orthant(grad_2nd, np.sign(pgrad_x))
        
        # do line search
        alpha = 1.
        while True:
            x_next = _constraint_orthant(x - grad_2nd_orthant * alpha, orthant)
            f_x_next, grad_x_next = eval_fun(x_next)
            g_x_next = _regulariser_fun(x_next)
            if f_x_next + g_x_next <= F_x + gamma * np.sum(pgrad_x * (x_next - x)): break
            alpha *= beta
            # n += 1
            # print(n)
        
        # check convergence
        if np.abs(f_x_next - f_x) / (1e-10 + np.abs(f_x)) < 1e-6: break
        n_iter += 1
        
        # update the histories
        S.append(x_next - x)
        Y.append(grad_x_next - grad_x)
        if len(S) > m: S.pop(0)
        if len(Y) > m: Y.pop(0)
    
    ############################### output ###############################
    if verbose_output:
        return {"x": x_next, "n_iter": n_iter, "fx": f_x_next, "gradx": grad_x_next, "gx": g_x_next}
    else:
        return x_next

def _get_initial_x(initial_x, input_size):
    if not hasattr(initial_x, "__iter__"):
        if initial_x == 0: # zeros initialisation
            initial_x = np.zeros(input_size)
        elif initial_x == 1: # random initialisation
            initial_x = np.random.random(input_size)
    return initial_x

def _get_regulariser(regulariser, regulariser_function, thresholding_function):
    if hasattr(regulariser_function, '__call__'): # if the regulariser fun is callable, then set the regulariser function as regulariser * regulariser_fun
        regulariser_fun = lambda x: regulariser * regulariser_function(x)
        if not hasattr(thresholding_function, '__call__'): raise ("Argument error: the thresholding_function argument must be a function")
        thresholding_fun = thresholding_function
        
    elif type(regulariser_function) == type("str"): # if regulariser function is specifid as a string
        regulariser_str = regulariser_function.lower()
        
        if regulariser_str == "l1":
            thresholding_fun = soft_threshold_l1
            regulariser_fun = lambda x: regulariser * reg_l1(x)
        
        elif regulariser_str == 'iso_tv' or regulariser_str == 'tv':
            thresholding_fun = soft_threshold_iso_tv
            regulariser_fun = lambda x: regulariser * reg_iso_tv(x)
        
        else:
            raise("Argument error: Unknown regulariser %s" % regulariser_str)
    
    return regulariser_fun, thresholding_fun
