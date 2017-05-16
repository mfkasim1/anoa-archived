################################### 2D TRANSFORMATIONS ###################################

class ParallelLineIntegral2D(ops.Transform):
    """
    ???
    """
    def __init__(self, angle, shape, num_pixel, centre=-1):
        ops.Transformation.__init__(self)
        self.angle = angle
        self.shape = shape
        self.num_pixel = num_pixel
        
        if centre == -1: self.centre = [a/2. for a in shape]
        else: self.centre = centre
    
    def forward(self, x):
        proj = np.array([])
        for i in range(self.num_pixel):
            d = -(self.num_pixel - 1)/2. + i
            S = _getPixelsCrossedByLine(x.shape, self.centre, d, self.angle)
            proj = np.append(proj, np.sum(S * x))
        
        return proj
    
    @make_output_array
    def adjoint(self, proj):
        # check input
        assert len(proj) != self.num_pixel, "length of the projection vector must be the same with the specified num_pixel"
        
        x = np.zeros(shape)
        for i in range(proj):
            d = -(self.num_pixel - 1)/2. + i
            S = _getPixelsCrossedByLine(self.shape, self.centre, d, self.angle)
            x += S * proj[i]
        
        return x

################################### MISC ###################################
# same as getPixelsCrossedByLine, but 0 <= np.tan(theta) <= 1
def _getPixelsCrossedByPosShallowLine(shape, centres, d, theta):
    Ny, Nx = shape
    cy, cx = centres
    X0, Y0 = np.meshgrid(np.arange(Nx)-cx, np.arange(Ny)-cy)
    tant = np.tan(theta)
    cost = np.cos(theta)
    sint = np.sin(theta)
    
    # escape division by zeros
    if (tant == 0): 
        return (d >= Y0) * (d < Y0+1)
    
    # compute the crossing points
    XL = X0
    YL = XL * tant + d / cost
    XR = X0 + 1
    YR = XR * tant + d / cost
    
    YB = Y0
    XB = YB / tant - d / sint
    YU = Y0 + 1
    XU = YU / tant - d / sint
    
    # check if the line comes from bottom, if so, then it must exit from the right
    comeFromBottom = np.logical_and(XB >= X0, XB < X0+1)
    XL[comeFromBottom] = XB[comeFromBottom]
    YL[comeFromBottom] = YB[comeFromBottom]
    
    # check if the line exit from the top, if so, then it must come from the left
    exitFromUp = np.logical_and(np.logical_not(comeFromBottom), np.logical_and(XU >= X0, XU < X0+1))
    XR[exitFromUp] = XU[exitFromUp]
    YR[exitFromUp] = YU[exitFromUp]
    
    # check if the pixel is crossed
    crossed = np.logical_and(np.logical_and(XL >= X0, XL < X0+1), np.logical_and(YL >= Y0, YL < Y0+1))
    weights = crossed * np.sqrt(np.square(XL - XR) + np.square(YL - YR))
    return weights

# get pixels crossed by line with the weights on shape (Ny x Nx) matrix with distance from the centre is d and form an angle theta in radians
def _getPixelsCrossedByLine(shape, centres, d, theta):
    # normalise the angle to have value between [0, pi)
    theta = np.mod(theta, np.pi)
    
    if (0 <= theta <= np.pi/4):
        weights = _getPixelsCrossedByPosShallowLine(shape, centres, d, theta)
        return weights
    elif (theta <= np.pi/2):
        weights = _getPixelsCrossedByPosShallowLine(shape, centres, d, np.pi/2-theta)
        return np.transpose(weights)
    elif (theta <= 3*np.pi/4):
        weights = _getPixelsCrossedByPosShallowLine(shape, centres, d, theta-np.pi/2)
        return np.transpose(weights)[:,::-1]
    else:
        weights = _getPixelsCrossedByPosShallowLine(shape, centres, d, np.pi-theta)
        return weights[::-1,:]

