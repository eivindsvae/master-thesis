import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from math import factorial
import cmath
from sympy.physics.quantum.cg import CG
from tqdm import tqdm
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 10})

# ALGORITHM

def Julia(center, range_, max_iterations, rv, rres, phires, remove=False):
    """
    INPUTS:
    center: (Complex) float c, for the polynomial z^2 + c
    range_: (Real) float, for deciding when an orbit "diverges"
    max_iterations: Int, for capping the amount of iterations per point if the orbit does not diverge
    rv: Float, the upper limit for absolute value of the complex numbers chosen to be studied
    rres: Int, the number of points across the interval (0, rv) to be studied
    phires: Int, the number of points across the interval (0, 2pi) to be studied
    remove: Boolean, to decide if the interior should be removed
    
    OUTPUTS:
    rcoordinates, phicoordinates: 1D arrays containing r- and phi-coordinates of all points
    jul: 3D array of the points that belong to the (filled) Julia set
    juliaSet: 2D array of all points, with their assigned value from the algorithm
    """
    # Creating the separations along axes, in polar coordinates
    dr, dph = rv/rres, 2*np.pi/phires
    # Creating arrays of r- and phi-coordinates for inserting into the polynomial
    rcoordinates = [(0 + dr*i) for i in range(rres+1)]  # r-coordinates of the sample-points
    phicoordinates = [(0 + dph*i) for i in range(phires+1)]  # phi-coordinates of the sample-points
    # Creating empty arrays to hold the final values
    juliapoints = []
    juliaSet = [[None for i in rcoordinates] for j in phicoordinates] # Creating 2d array with None-values, of size rcoordinates*phicoordinates
    dictset = [] # An intermediate list to hold values before removing the interior
    Juldict = {} # A dictionary which is used to keep count of the indexes for the points, used to remove the interior
    # Replaces each None-value with the number of iterations needed for the corresponding complex point to escape outside range_
    # If it never crosses range_, the value is set to 0
    for ph in range(len(phicoordinates)):
        for r in range(len(rcoordinates)):
            # Defining the complex number, both on rectangular form and using polar coordinates of phase and amplitude
            z = cmath.rect(rcoordinates[r], phicoordinates[ph])
            radi = rcoordinates[r]
            angu = phicoordinates[ph]    
            iteration = 0 # Setting an iteration counter
            while(abs(z) < range_ and iteration < max_iterations): # Repeat until orbit escapes outside range_, or maximum iterations is reached
                iteration += 1 # Creates a value where "how fast" the orbit escapes outside range_ is stored (only useful for plots)
                z = z**2 + center # Applying the polynomial to the chosen complex point
            if(iteration == max_iterations):
                juliaSet[ph][r] = 0 # Complex point belongs to the Julia set
                Juldict[(radi, angu)] = (r, ph) # Adding the indices to the dictionary for later removal of the interior
                dictset.append(np.array([radi, angu]))
            else:
                juliaSet[ph][r] = iteration # Complex point doesn't belong to the Julia set (value is only useful for plots)
    #If the Julia set needs its interior removed, the following if-statement removes any point where all neighbouring points are also in the set
    if remove:
        # Retrieve the dictionary values as a set
        l = set(Juldict.values())
        # A function to handle the indexes of phi-coordinates at 0 and 2pi
        def borderphi(phi_index):
            # For the "negative" neighbours of phi = 0, return the index for phi = 2pi
            if phi_index < 0:
                return phires
            # For the "positive" neighbours of phi = 2pi, return the index for phi = 0
            elif phi_index > phires:
                return 0
            # For all other values, do nothing
            else:
                return phi_index
            
        # Define another list to hold the values that are kept while the old one is iterated over
        dictset2 = []
        # Iterate through all elements in the filled Julia set as keys in Juldict
        for elt in dictset:
            key = tuple(elt)
            # Adding in a safety in case of roundoff errors or other nonsense
            if key not in Juldict:
                continue
            # Retrieving the dictionary value, i.e. the indices in the original polar grid for a point in the filled Julia set
            a, b = Juldict[key]
            # Creating all eight neighbouring points on the grid, using borderphi to take care of the phi-coordinate in the case of phi = 0 or 2pi
            neighbours = [(a+1, b), (a-1, b), (a, borderphi(b+1)), (a, borderphi(b-1)), (a+1, borderphi(b+1)), (a+1, borderphi(b-1)), (a-1, borderphi(b+1)), (a-1, borderphi(b-1))]
            # Making sure all neighbours have valid r-coordinate indexing by ignoring negative values or values on the edge of the original grid
            valid_neighbours = [(nr, np) for nr, np in neighbours if 0 <= nr <= rres]
            # If all neighbouring points on the original grid are also in dictset, the point is removed
            if all(neigh in l for neigh in valid_neighbours):
                continue
            # If not, add them to dictset2
            dictset2.append(elt)
        dictset = dictset2
    # Now, expand each point into rectangles. Iterate over all points in the Julia set
    for elt in dictset:
        radial = elt[0]
        angular = elt[1]
        # Create a "rectangle" around each point, for later integration purposes
        R_square = np.array([max(0, radial - dr/2), max(0, radial - dr/2), min(rv, radial + dr/2), min(rv, radial + dr/2)])
        Phi_square = np.array([min(2*np.pi, angular + dph/2), max(0, angular - dph/2), max(0, angular - dph/2), min(2*np.pi, angular + dph/2)])
        # Adding the square to the result array, in the proper format
        juliapoints.append(np.array([R_square, Phi_square]))
    jul = np.array(juliapoints)
    return rcoordinates, phicoordinates, jul, juliaSet

def RiemannFractal(center, range_, max_iterations, rv, rres, phires, remove=False):
    """
    INPUTS:
    center: (Complex) float c, for the polynomial z^2 + c
    range_: (Real) float, for deciding when an orbit "diverges"
    max_iterations: Int, for capping the amount of iterations per point if the orbit does not diverge
    rv: Float, the upper limit for absolute value of the complex numbers chosen to be studied
    rres: Int, the number of points across the interval (0, rv) to be studied
    phires: Int, the number of points across the interval (0, 2pi) to be studied
    remove: Boolean, to decide if the interior should be removed
    
    OUTPUTS:
    xs, ys: 1D arrays of all the r- and phi-values in our resolution, regardless of whether or not they belong to the Julia set, for plotting
    rfrac: 3D array of the points that belong to the (filled) Julia set, expressed in their spherical coordinates on the Riemann sphere
    colours: 2D array of all assigned values from the algorithm, for plotting
    """
    # Mapping the points in the Julia set onto the Riemann sphere through stereographic projection, returning coordinates (theta, phi)
    # First, determining which points to consider by using the function Julia()
    xs, ys, pointset, colours = Julia(center, range_, max_iterations, rv, rres, phires, remove)
    # Separating the complex numbers into real and imaginary coorinates a and b
    a_vals = pointset[:, 0, :]
    b_vals = pointset[:, 1, :]
    # Generating an array of zeros for the final coordinates to replace
    N = len(a_vals)
    rfrac = np.zeros(shape=(N, 2, 4))
    for i in range(N):
        # Choosing one square
        A_square = a_vals[i]
        B_square = b_vals[i]
        for n in range(4):
            # Choosing one point
            A = A_square[n]
            B = B_square[n]
            # Performing stereographic projection in polar coordinates of the point
            if A == 0:
                theta = np.pi
            else:
                theta = 2*np.arctan(1/A)
            phi = B
            # Taking care to yield theta values in the interval [0, pi]
            if theta < 0:
                theta += 2*np.pi
            # Replacing the zeros in the correct positions of the final array
            rfrac[i, :, n] = [theta, phi]
    return xs, ys, rfrac, colours

def phire(phi, mprime):
    """
    Input:
    phi: float, the angle our state is rotated to, 0 < phi < 2*pi
    mprime: float, the m our state is rotated to, -s < mprime < s
    Output:
    The real part of the phi-dependent factor in Wigners D-matrix elements
    """
    return np.cos(mprime*phi)

def phiim(phi, mprime):
    """
    Input:
    phi: float, the angle our state is rotated to, 0 < phi < 2*pi
    mprime: float, the m our state is rotated to, -s < mprime < s
    Output:
    The imaginary part of the phi-dependent factor in Wigners D-matrix elements
    """
    return -np.sin(mprime*phi)

def constantfunc(k, s, mprime):
    """
    Input:
    k: int, the value for the summing variable
    s: float, the spin number of the rotated state
    mprime: float, the m our state is rotated to, -s < mprime < s
    Output:
    The factor in Wigners D-matrix elements which is independent of angles phi and theta
    """
    return ((-1)**(k - s + mprime))*(np.sqrt(float(factorial(int(2*s))))*np.sqrt(float(factorial(int(s + mprime))))*np.sqrt(float(factorial(int(s - mprime))))/(float(factorial(k))*float(factorial(int(2*s - k)))*float(factorial(int(s - k - mprime)))*float(factorial(int(k - s + mprime)))))

def thetaf(theta, k, s, mprime):
    """
    Input:
    theta: float, the angle our state is rotated to, 0 < theta < pi
    k: int, the value for the summing variable
    s: float, the spin number of the rotated state
    mprime: float, the m our state is rotated to, -s < mprime < s
    Output:
    The theta-dependent factor in Wigners D-matrix elements (including sin(theta) from the integration measure)
    """
    return np.sin(theta)*((np.cos(theta/2))**(3*s - 2*k - mprime))*((np.sin(theta/2))**(2*k - s + mprime))

def integral(N, fractal):
    """
    Input:
    N: int, the number of particles in the system
    fractal: 2D array, the fractal generated to integrate over
    Output:
    totstate: 1D array, the total state
    """
    # Define s from N, and set an empty array to hold the state
    s = N/2
    ints = np.zeros(N + 1)
    intelts = ints.astype(complex)
    # Iterate over all basis elements for the total state
    for i in tqdm(range(N + 1)):
        # Set mprime for this basis element and create an empty array to hold contributions from all parts of the fractal
        mprime = -s + i
        ns = np.zeros(len(fractal))
        nsum = ns.astype(complex)
        # Iterate over all pieces of the fractal
        for n in range(len(fractal)):
            # Set the integration limits and create an empty array to hold contributions from all the summands
            thetamin = min(fractal[n][0][0], fractal[n][0][2])
            thetamax = max(fractal[n][0][0], fractal[n][0][2])
            phimin = min(fractal[n][1][0], fractal[n][1][2])
            phimax = max(fractal[n][1][0], fractal[n][1][2])
            ks = np.zeros(N + 1)
            ksum = ks.astype(complex)
            # Iterate over the sum in Wigners D-matrix elements
            for k in range(N + 1):
                # Verify that the given k is a valid summing value (i.e. leads to factorials being defined)
                if s - k - mprime >= 0 and k - s + mprime >= 0:
                    # Calculate the constant factor in the Wigner D-matrix elements, and perform the integral over theta
                    con = constantfunc(k, s, mprime)
                    the = quad(thetaf, thetamin, thetamax, args=(k, s, mprime))[0]
                    # Perform the integral over phi of the real and imaginary part separately
                    rephi = quad(phire, phimin, phimax, args=(mprime))[0]
                    imphi = quad(phiim, phimin, phimax, args=(mprime))[0]
                    # Multiply the separated integrals together
                    re = rephi*the
                    im = imphi*the
                    # Add the result up to one complex number, making sure to include the constant
                    ksum[k] = con*(re + im*1j)
            # Sum all the contributions from all terms in the k-sum
            nsum[n] = np.sum(ksum)
        # Sum all the contributions from all pieces of the fractal
        intelts[i] = np.sum(nsum)
    # Calculate the norm squared of the integrated array as a vector
    normalize = sum(abs(intelts)**2)
    # Divide by the norm to normalize the state
    totstate = intelts/np.sqrt(normalize)
    return totstate

def eigentoproduct(syststate, subsize):
    """
    Input:
    syststate: 1D array, the coefficients for our state in the standard Hilbert space basis
    subsize: int, the amount of particles from our system which are considered the subsystem
    Output:
    newstate: 1D array, the coefficients for our state in the Schmidt decomposed basis
    """
    # Similarly as before, define system size Ntot and total spin stot
    Ntot = len(syststate) - 1
    stot = Ntot/2
    # Define an empty array of the correct size to hold coefficients for the new basis
    news = np.zeros((Ntot - int(subsize) + 1)*(int(subsize) + 1))
    newstate = news.astype(complex)
    # Define a variable to keep count on what element in the array to fill next
    fillcount = 0
    # Define spins s1 for the subsystem and s2 for the rest of the system
    s1 = subsize/2
    s2 = (Ntot - subsize)/2
    # Iterate over all values of m1 from -s1 to s1
    for n in range(int(subsize) + 1):
        m1 = -s1 + n
        # Iterate over all values for mtot from -s to s
        for i in range(Ntot + 1):
            # Define mtot, pick out a coefficient from the standard Hilbert space basis and define m2
            mtot = -stot + i
            cons = syststate[i]
            m2 = mtot - m1
            # Check whether or not the suggested state is even possible (i.e. -s2 < m2 < s2)
            if m2 <= s2 and m2 >= -s2:
                # Fill each element of the Schmidt decomposed basis with our coefficient multiplied with the correct Clebsch-Gordan coefficient
                newstate[fillcount] = cons*complex(CG(s1, m1, s2, mtot - m1, stot, mtot).doit().evalf(), 0)
                # Update the fillcount to ensure the next valid coefficient gets placed correctly
                fillcount += 1
    return newstate

def densitymatrix(integralstate, subsize):
    """
    Input:
    integralstate: 1D array, the state given by the generated fractal, integrated over the Riemann-sphere
    Output:
    densemat: 2D array, the total density matrix of our system, in the Schmidt decomposed basis
    """
    # Get the state in the Schmidt decomposed basis
    state = eigentoproduct(integralstate, subsize)
    # Define the bra by conjugating the coefficients
    constate = np.conjugate(state)
    # Create the density matrix by using the outer product
    densemat = np.outer(state, constate)
    return densemat

def reddensmat(N, sssize, densmat):
    """
    Input:
    N: int, total system size
    sssize: int, the size of the subsystem considered
    densmat: 2D array, the total density matrix for the system given by our fractal
    Output:
    evals: 1D array, the eigenvalues of the reduced density matrix for the subsystem in the given fractal state
    """
    # Defining identity matrices, one for the subsystem, and one to obtain unit vectors for the remainder of the system
    idA = np.eye(int(sssize) + 1)[np.newaxis]
    basisB = np.eye(N - int(sssize) + 1)[:,:,np.newaxis]
    # Creating basis of kets formed by the Kronecker products of IdA and basisB-unit vectors, transposing to obtain brabasis
    ketbasis = np.kron(idA, basisB)
    brabasis = np.transpose(ketbasis, [0,2,1])
    # Computing the matrix products with the density matrix, summing the contributions for the different basis vectors
    ket = np.matmul(densmat,ketbasis)
    rdmat = np.sum(np.matmul(brabasis, ket), 0)
    # Calculating the eigenvalues of the total reduced density matrix
    evals = np.linalg.eigvals(rdmat)
    # Ordering the eigenvalues from large to small real parts
    x = np.argsort(evals)
    evals = evals[x]
    evals = evals[::-1]
    return evals

def vonNeumann(eigs):
    """
    Input:
    eigs: 1D array, the eigenvalues of the reduced density matrix of a subsystem in a state given by our fractal
    Output:
    entropy: float, the entanglement entropy for the subsystem with the system in the given fractal state
    """
    # The formula for von Neumann entropy expressed in eigenvalues: -Σ(λ*ln(λ))
    # Set a variable to hold the value of the trace so far
    trace = 0
    # Create an array to hold elements of the sum
    entent = np.zeros(len(eigs))
    # Iterate over all eigenvalues
    for i in range(len(eigs)):
        # Check if the trace currently is decently close to 1
        if np.real(trace) >= 0.999999999:
            # If it is, the remaining eigenvalues are small enough to neglect, and we use that xlog(x) tends to 0 as x tends to 0
            entent[i] = 0
        else:
            # If not, calculate xlog(x)
            entent[i] = np.real(eigs[i])*np.log(np.real(eigs[i]))
        # Update the current trace
        trace += eigs[i]
    # Finally, sum all the contributions and switch sign
    entropy = -np.sum(entent)
    return entropy

# FUNCTION TO FULLY PERFORM PROCEDURE FROM START TO FINISH, GIVEN ALL PARAMETERS
def EntanglementEntropy(center, range_, max_iterations, rv, rres, phires, remove, N, sizes):
    """
    INPUTS:
    center: (Complex) float c, for the polynomial z^2 + c
    range_: (Real) float, for deciding when an orbit "diverges"
    max_iterations: Int, for capping the amount of iterations per point if the orbit does not diverge
    rv: Float, the upper limit for absolute value of the complex numbers chosen to be studied
    rres: Int, the number of points across the interval (0, rv) to be studied
    phires: Int, the number of points across the interval (0, 2pi) to be studied
    remove: Boolean, to decide if the interior should be removed
    N: int, total system size
    sizes: 1D array of all subsystem sizes in ascending order
    
    OUTPUTS:
    xs, ys: 1D arrays of all the x- and y-values in our resolution, regardless of whether or not they belong to the Julia set
    ps: 2D array of all points, with their assigned value from the algorithm, for plotting
    ee: 1D array of the entanglement entropies of subsystems of all chosen values in sizes
    """
    # First generating the fractal, both the heatmap values for plotting and the stereographically projected theta- and phi-values
    xs, ys, vs, ps = RiemannFractal(center, range_, max_iterations, rv, rres, phires, remove)
    # Computing the state by integrating over the fractal
    intstate = integral(N, vs)
    # Creating an empty array to hold entanglement entropy values for different sized subsystems, as given in sizes
    ee = np.zeros(len(sizes))
    # Iterating over subsystem sizes
    for i, size in enumerate(sizes):
        # Computing the density matrix in the Schmidt decomposed basis, then computing the partial trace to get the reduced density matrix
        totdensmat = densitymatrix(intstate, size)
        ss = reddensmat(N, size, totdensmat)
        # Compute the von Neumann entropy
        ee[i] = vonNeumann(ss)
    return xs, ys, ps, sizes, ee

def cfit(n, d, s0):
    """
    Input:
    n: int, the variable for the function
    d: float, determining the prefactor in the function
    s0: float, a constant term
    Output:
    The function to fit the data curve to
    """
    return (d/2)*np.log(n - (n**2)/120) + s0

# ADJUSTABLE VARIABLES/PARAMETERS

# Setting c for the polynomial f(z) = z^2 + c
c = 0.32 + 0.043j
# Setting the range so that coordinates exceeding this range are considered diverging under successive application of f(z)
ran = 2#0.9 + abs(c) 
# Setting the maximum amount of successive applications of f(z) to a complex point before giving up
max_it = 500
# Setting radius of the sample area
r_val = 2
# Setting the resolution
rResolution = 1000
phiResolution = 1000
# Removing the interior
rem=True
# Some choices for subsystem size arrays
choice1 = np.linspace(4, 116, 29)
choice2 = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116])
choice3 = np.array([10, 20, 30, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 90, 100, 110])

# CALCULATION

xes, yes, fr, siz, ents = EntanglementEntropy(c, ran, max_it, r_val, rResolution, phiResolution, rem, 120, choice3)
pa, pr = curve_fit(cfit, siz, ents)
print("The entanglement entropy is:", ents)

# PLOTTING

# Plotting the entanglement entropy as a function of subsystem-size
plt.figure(figsize=(12,8))
#plt.title('WIIII')
plt.plot(siz, ents, 'o', color='#003f5c', label = 'Entanglement entropy for Julia states, L = 120') # Creating the graph
plt.plot(siz, cfit(siz, *pa), color='#bc5090', label = 'Fit: d=%5.3f, s0=%5.3f' % tuple(pa)) # Plotting the fitted curve and displaying the parameters

plt.xlabel(r'$n$', size=12) # Setting axis label
plt.ylabel(r'$S(n)$', size=12) # Setting axis label
plt.tight_layout()
plt.grid()
plt.legend(fontsize=16) # Displaying graph labels
plt.savefig("entent0320043i.png", dpi=300, bbox_inches='tight') # Saving the figure locally
plt.show()

# Plotting the fractal

# Convert polar grid (ae, be) to meshgrid and rewrite to Cartesian coordinates
#Rplots, Phiplots = np.meshgrid(xes, yes)
#X = Rplots * np.cos(Phiplots)
#Y = Rplots * np.sin(Phiplots)
# The plot itself
#plt.figure(figsize=(12,8))
#ax = plt.axes()
#ax.set_aspect('equal')    # Setting axes
#plot = ax.pcolormesh(X, Y, fr, cmap = 'magma')   # Creating the heatmap
#plt.colorbar(plot)        # Adding colourbar
#plt.tight_layout()
#plt.title('Julia-set \ncenter = {}, range = {:.3f}, max-iterations = {}'.format(c, ran, max_it))  # Setting title
#plt.savefig("julia_set_beauty2.png", dpi=300, bbox_inches='tight') # Saving the figure locally
#plt.show()