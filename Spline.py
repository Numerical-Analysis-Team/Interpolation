import numpy as np


def cubic_spline_interpolation_numpy(x_values, y_values, target_x):

    n = len(x_values) - 1
    h = np.diff(x_values)

    # Calculate alpha
    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (y_values[i + 1] - y_values[i]) - (3 / h[i - 1]) * (y_values[i] - y_values[i - 1])

    # Build tridiagonal matrix components
    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)

    for i in range(1, n):
        l[i] = 2 * (x_values[i + 1] - x_values[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # Calculate coefficients
    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_values[j + 1] - y_values[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # Compute the interpolated value at target_x
    for i in range(n):
        if x_values[i] <= target_x <= x_values[i + 1]:
            dx = target_x - x_values[i]
            result = y_values[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
            return result

    raise ValueError("The target_x value is outside the interpolation range.")


# Main function
def main():

    x_values = np.array([1, 2, 3, 4, 5])  # X values
    y_values = np.array([2, 8, 18, 32, 50])  # Y values


    target_x = 2.5


    result = cubic_spline_interpolation_numpy(x_values, y_values, target_x)


    print(f"The interpolated value at x = {target_x} using cubic spline is: {result:.4f}")



if __name__ == "__main__":
    main()
