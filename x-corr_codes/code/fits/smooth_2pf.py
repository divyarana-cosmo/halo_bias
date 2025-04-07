import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import legendre, eval_legendre
from scipy.interpolate import interp1d
from scipy.integrate import quad

def main():
    # Set the resolution of the HEALPix map
    nside = 128
    npix = hp.nside2npix(nside)
    lmax = 3 * nside - 1  # Maximum multipole for decomposition

    # Create a function of both theta and phi
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    
    # Example function f(theta, phi)
    f_map = np.sin(4*theta) * np.cos(2*phi) + 0.5*np.sin(theta) * np.sin(5*phi)
    
    # Create a function that depends only on theta
    g_theta = np.exp(-8 * (theta - np.pi/2)**2)
    
    # Visualize the original maps
    plt.figure(figsize=(12, 5))    
    plt.subplot(121)
    hp.mollview(f_map, title="Original f(theta, phi)", hold=True)    
    plt.subplot(122)
    hp.mollview(g_theta, title="Kernel g(theta)", hold=True)    
    plt.tight_layout()
    plt.savefig("original_maps.png")
    
    # Standard convolution
    convolved_map = convolve_spherical_functions(f_map, g_theta, nside, lmax)
    
    # Disc averaging
    theta_max = np.pi/12  # 15 degrees
    disc_averaged_map = disc_average_map(convolved_map, theta_max, nside)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    hp.mollview(convolved_map, title="Convolved Result f*g", hold=True)
    plt.subplot(122)
    hp.mollview(disc_averaged_map, title=f"Disc Averaged (θmax = {np.degrees(theta_max):.1f}°)", hold=True)
    plt.tight_layout()
    plt.savefig("convolution_results.png")
    
    return f_map, g_theta, convolved_map, disc_averaged_map

def convolve_spherical_functions(f_map, g_theta, nside, lmax):
    """
    Convolve a function f(theta,phi) with an axisymmetric function g(theta)
    using spherical harmonic decomposition.
    """
    # Get theta values for the map
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    
    # Convert f(theta, phi) to spherical harmonic coefficients
    f_alm = hp.map2alm(f_map, lmax=lmax)
    
    # Compute g_ell coefficients
    g_ell = compute_g_ell_coefficients_vectorized(g_theta, theta, lmax)
    
    # Apply convolution in harmonic space
    f_conv_alm = apply_convolution_vectorized(f_alm, g_ell, lmax)
    
    # Convert back to real space
    convolved_map = hp.alm2map(f_conv_alm, nside)
    
    return convolved_map

def compute_g_ell_coefficients_vectorized(g_theta, theta, lmax):
    """
    Vectorized computation of the multipole coefficients g_ell.
    """
    g_ell = np.zeros(lmax + 1)
    cos_theta = np.cos(theta)
    sin_theta_weights = np.sin(theta)
    weighted_g = g_theta * sin_theta_weights
    
    for ell in range(lmax + 1):
        p_ell_values = legendre(ell)(cos_theta)
        integrand = weighted_g * p_ell_values
        g_ell[ell] = ((2 * ell + 1) / 2) * np.trapz(integrand, x=theta)
    
    return g_ell

def apply_convolution_vectorized(f_alm, g_ell, lmax):
    """
    Apply convolution in spherical harmonic space.
    """
    f_conv_alm = np.zeros_like(f_alm, dtype=complex)
    ell_values = np.arange(lmax + 1)
    scale_factors = np.sqrt(4 * np.pi / (2 * ell_values + 1)) * g_ell
    
    for ell in range(lmax + 1):
        scale = scale_factors[ell]
        for m in range(-ell, ell + 1):
            idx = hp.Alm.getidx(lmax, ell, abs(m))
            f_conv_alm[idx] = f_alm[idx] * scale
    
    return f_conv_alm

def disc_average_map(input_map, theta_max, nside):
    """
    Average the map over discs of angular radius theta_max centered at each pixel.
    
    Parameters:
    -----------
    input_map : array-like
        Input HEALPix map to be averaged
    theta_max : float
        Maximum angular radius (in radians) for the averaging disc
    nside : int
        HEALPix nside parameter
        
    Returns:
    --------
    disc_averaged_map : array-like
        The input map averaged over discs
    """
    npix = hp.nside2npix(nside)
    disc_averaged_map = np.zeros_like(input_map)
    
    # For each pixel, find neighbors within theta_max and average
    for i in range(npix):
        # Get the pixel's coordinates
        vec = hp.pix2vec(nside, i)
        
        # Find all pixels within theta_max
        disc_pixels = hp.query_disc(nside, vec, theta_max)
        
        # Average the values of these pixels
        disc_averaged_map[i] = np.mean(input_map[disc_pixels])
    
    return disc_averaged_map

def disc_average_map_vectorized(input_map, theta_max, nside):
    """
    More optimized version of disc averaging that reduces redundant calculations.
    Uses harmonic space for efficiency when theta_max is large.
    
    This approach is better for larger theta_max values:
    - For small theta_max: use direct pixel-based approach
    - For large theta_max: use beam smoothing in harmonic space
    """
    # For large angular scales, using harmonic smoothing is more efficient
    if theta_max > np.radians(5):  # arbitrary threshold, adjust based on your needs
        # Convert theta_max to FWHM for beam smoothing
        # The relationship is approximately FWHM ≈ 2.355 * sigma
        # And for a top-hat disc of radius theta_max, sigma ≈ theta_max / sqrt(2)
        fwhm = 2.355 * theta_max / np.sqrt(2)
        
        # Smooth using spherical harmonic transform
        lmax = 3 * nside - 1
        alm = hp.map2alm(input_map, lmax=lmax)
        smoothed_alm = hp.smoothalm(alm, fwhm=fwhm, inplace=False)
        return hp.alm2map(smoothed_alm, nside)
    else:
        # For small theta_max, use the direct pixel-based approach
        return disc_average_map(input_map, theta_max, nside)

def main_with_options():
    """
    Extended main function with various options and demonstration of different methods.
    """
    # Set the resolution of the HEALPix map
    nside = 128
    npix = hp.nside2npix(nside)
    lmax = 3 * nside - 1  # Maximum multipole for decomposition

    # Create example functions
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    f_map = np.sin(4*theta) * np.cos(2*phi) + 0.5*np.sin(theta) * np.sin(5*phi)
    g_theta = np.exp(-8 * (theta - np.pi/2)**2)
    
    # Standard convolution
    convolved_map = convolve_spherical_functions(f_map, g_theta, nside, lmax)
    
    # Try different disc averaging methods and theta_max values
    theta_max_values = [np.pi/30, np.pi/15, np.pi/10]  # 6°, 12°, 18°
    
    # Create plots for different theta_max
    plt.figure(figsize=(15, 10))
    
    # Plot original convolved map
    plt.subplot(2, 2, 1)
    hp.mollview(convolved_map, title="Convolved Map (No Averaging)", hold=True)
    
    # Plot disc averaged maps for different theta_max values
    for i, theta_max in enumerate(theta_max_values):
        # Use the vectorized disc averaging function
        avg_map = disc_average_map_vectorized(convolved_map, theta_max, nside)
        
        plt.subplot(2, 2, i+2)
        hp.mollview(avg_map, 
                   title=f"Disc Avg (θmax = {np.degrees(theta_max):.1f}°)", 
                   hold=True)
    
    plt.tight_layout()
    plt.savefig("disc_averaging_comparison.png")
    
    return convolved_map, theta_max_values

if __name__ == "__main__":
    main()
