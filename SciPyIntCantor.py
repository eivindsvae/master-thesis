import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
from math import factorial
from sympy.physics.quantum.cg import CG
from tqdm import tqdm
from scipy.optimize import curve_fit

# ALGORITHM

def Cantor(N, r, k):
    """
    Input:
    N: int, amount of intervals to keep from one previous interval in the next iteration
    r: float, the fractional size of each interval compared to the intervals in the previous iteration of the Cantor set
    k: int, number of iterations in generating the Cantor set
    Output:
    cantorset: 2D array, start- and endpoints of the intervals approximating the Cantor set
    """
    # Setting the start C(N, r, 0)
    cantorset = np.array([[0, 1]])
    # Iterating up to k
    for i in range(k):
        # Creating an empty array to hold the Cantor set
        newcantor = list(np.zeros(N**(i + 1)))
        # Iterating over the elements in the previous Cantor set
        for n in range(len(cantorset)):
            # Defining the length of an interval in this iteration
            interlength = (cantorset[n][1] - cantorset[n][0])/(1/r)
            # Setting a counter for which interval is to be determined next
            mcounter = 0
            # Iterating over all the 1/r intervals an element in the previous iteration is now split into
            for m in range(int(1/r)): #NOT PERFECT; ASSUMES r = 1/(2N - 1)
                # Verifying that this interval is to be kept in the next iteration
                if m % 2 == 0:
                    # Setting the valid interval in newcantor
                    newcantor[N*n + mcounter] = np.array([cantorset[n][0] + m*interlength, cantorset[n][0] + (m + 1)*interlength])
                    # Updating the counter
                    mcounter +=1
        cantorset = np.array(newcantor)
    return cantorset

def RiemannCantor(N, r, k):
    """
    Input:
    N: int, amount of intervals to keep from one previous interval in the next iteration
    r: float, the fractional size of each interval compared to the intervals in the previous iteration of the Cantor set
    k: int, number of iterations in generating the Cantor set
    Output:
    Riefrac: 2D array, phi-coordinates of the intervals approximating the Cantor set, projected onto the equator of the Riemann sphere
    """
    # Generating the Cantor set
    frac = Cantor(N, r, k)
    # Creating an empty array to hold the projected phi-values
    Riefrac = np.zeros(frac.shape)
    # Iterating over all intervals
    for i in range(len(frac)):
        # Setting both the start- and endpoint as 2*pi multiplied by the start- and endpoints of the Cantor set, thus projecting it onto the Riemann sphere
        Riefrac[i][0] = 2*np.pi*frac[i][0]
        Riefrac[i][1] = 2*np.pi*frac[i][1]
    return Riefrac

def ReCantorWig(phi, k, s, mprime):
    """
    Input:
    phi: float, the angle for a given rotated state
    k: int, the summing variable in the expression for Wigners D-matrix elements
    s: float, the total spin of the state we are considering (i.e. N/2)
    mprime: float, the spin number of the basis element we are considering
    Output:
    float, the real part of one sum element in an element of Wigners D-matrix
    """
    return np.cos(mprime*phi)*(-1)**(k - s + mprime)*(np.sqrt(float(factorial(int(2*s))))*np.sqrt(float(factorial(int(s + mprime))))*np.sqrt(float(factorial(int(s - mprime))))/(float(factorial(k))*float(factorial(int(2*s - k)))*float(factorial(int(s - k - mprime)))*float(factorial(int(k - s + mprime)))))*0.5**s

def ImCantorWig(phi, k, s, mprime):
    """
    Input:
    phi: float, the angle for a given rotated state
    k: int, the summing variable in the expression for Wigners D-matrix elements
    s: float, the total spin of the state we are considering (i.e. N/2)
    mprime: float, the spin number of the basis element we are considering
    Output:
    float, the imaginary part of one sum element in an element of Wigners D-matrix
    """
    return -np.sin(mprime*phi)*(-1)**(k - s + mprime)*(np.sqrt(float(factorial(int(2*s))))*np.sqrt(float(factorial(int(s + mprime))))*np.sqrt(float(factorial(int(s - mprime))))/(float(factorial(k))*float(factorial(int(2*s - k)))*float(factorial(int(s - k - mprime)))*float(factorial(int(k - s + mprime)))))*0.5**s
    
def integral1d(N, fractal):
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
            phimin = min(fractal[n][0], fractal[n][1])
            phimax = max(fractal[n][0], fractal[n][1])
            ks = np.zeros(N + 1)
            ksum = ks.astype(complex)
            # Iterate over the sum in Wigners D-matrix elements
            for k in range(N + 1):
                # Verify that the given k is a valid summing value (i.e. leads to factorials being defined)
                if s - k - mprime >= 0 and k - s + mprime >= 0:
                    # Perform the integral of the real and imaginary part separately
                    re = quad(ReCantorWig, phimin, phimax, args=(k, s, mprime))[0]
                    im = quad(ImCantorWig, phimin, phimax, args=(k, s, mprime))[0]
                    # Add the result up to one complex number
                    ksum[k] = re + im*1j
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
def EntanglementEntropy(n, r, k, N, sizes):
    """
    INPUTS:
    n: int, amount of intervals to keep from one previous interval in the next iteration
    r: float, the fractional size of each interval compared to the intervals in the previous iteration of the Cantor set
    k: int, number of iterations in generating the Cantor set
    N: int, total system size
    sizes: 1D array of all subsystem sizes in ascending order
    
    OUTPUTS:
    xs, ys: 1D arrays of all the x- and y-values in our resolution, regardless of whether or not they belong to the Julia set
    ps: 2D array of all points, with their assigned value from the algorithm, for plotting
    ee: 1D array of the entanglement entropies of subsystems of all chosen values in sizes
    """
    # First generating the fractal, both the heatmap values for plotting and the stereographically projected theta- and phi-values
    vs = RiemannCantor(n, r, k)
    # Computing the state by integrating over the fractal
    intstate = integral1d(N, vs)
    # Creating an empty array to hold entanglement entropy values for different sized subsystems, as given in sizes
    ee = np.zeros(len(sizes))
    # Iterating over subsystem sizes
    for i in tqdm(range(len(sizes))):
        # Computing the density matrix in the Schmidt decomposed basis, then computing the partial trace to get the reduced density matrix
        totdensmat = densitymatrix(intstate, sizes[i])
        ss = reddensmat(N, sizes[i], totdensmat)
        # Compute the von Neumann entropy
        ee[i] = vonNeumann(ss)
    return sizes, ee

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

# CALCULATION

Totalsize = 120
a, b = EntanglementEntropy(2, 1/3, 15, Totalsize, np.linspace(20, 100, 17))#np.linspace(2, 118, 30))np.linspace(1, 119, 60))
pa, pr = curve_fit(cfit, a, b)
print("The entanglement entropy is:", b)

# PLOTTING

# Plotting the entanglement entropy as a function of subsystem-size
plt.figure(figsize=(12,8))
plt.plot(a, b, 'o', color='#003f5c', label = 'Entanglement entropy for Cantor states, N = %5.3f' % Totalsize) # Creating the graph
plt.plot(a, cfit(a, *pa), color='#bc5090', label = 'Fit: d=%5.3f, s0=%5.3f' % tuple(pa)) # Plotting the fitted curve and displaying the parameters
plt.xlabel(r'$n$', size=12) # Setting axis label
plt.ylabel(r'$S(n)$', size=12) # Setting axis label
plt.tight_layout()
plt.grid()
plt.legend() # Displaying graph labels
plt.savefig("cantor_entent.png", dpi=300, bbox_inches='tight') # Saving the figure locally
plt.show()