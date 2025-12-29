def sample1():
    
    """ Simple 10 point shit """
    
    # Data Block

    X = [1  ,2   ,3   ,4   ,5   ,6   ,7   ,8   ,9   ,10]
    Y = [1.5,13.7 ,3.2 ,5   ,4.1 ,7   ,5.6 ,13  ,11  ,9 ]

    # Combine the data into a DataFrame
    og_data = pd.DataFrame({'X': X, 'Y': Y})
    s_row, s_col = og_data.shape
    cols = list(og_data.columns)

    # Print the DataFrame
    print(og_data)
    print(f'\nNo of samples : {s_row}\nNo of features(dimensions) : {s_col}')

    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    #   Create a scatter plot
    plt.scatter(og_data['X'], og_data['Y'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.title('Scatter plot of original data : X and Y')
    plt.show()

    return og_data
    

def sample2():

    """ Unevenly spread through out the margins """

    # set the random seed for reproducibility
    np.random.seed(123)

    # generate a random dataset with 2 columns and 100 samples
    X = np.random.rand(100, 2)

    # print the first 5 rows of the dataset
    print(X)

    og_data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1]})
    s_row, s_col = og_data.shape
    cols = list(og_data.columns)

    # Print the DataFrame
    print(og_data)
    print(f'\nNo of samples : {s_row}\nNo of features(dimensions) : {s_col}')

    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot
    plt.scatter(og_data['X'], og_data['Y'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.title('Scatter plot of original data : X and Y')
    plt.show()
    
    return og_data
    return s_row
    return s_col
    return cols

def sample3():
    
    """ centre cluster with uneven point around the cluster """
    
    # set the random seed for reproducibility
    np.random.seed(123)

    # generate a random dataset with 2 columns and 100 samples
    X = np.random.rand(100, 2)

    a = 2.0 # major axis
    b = 1.0 # minor axis
    theta = np.random.rand(100) * 2.0 * np.pi
    r = np.sqrt(np.random.rand(100))
    x = a * r * np.cos(theta)
    y = b * r * np.sin(theta)

    # generate 20 outlier points
    outliers_x = np.random.uniform(low=-5, high=5, size=20)
    outliers_y = np.random.uniform(low=-5, high=5, size=20)

    # combine the x and y coordinates of the ellipse and outliers into a 2-column array
    X = np.vstack((np.column_stack((x, y)), np.column_stack((outliers_x, outliers_y))))

    og_data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1]})
    s_row, s_col = og_data.shape
    cols = list(og_data.columns)

    # Print the DataFrame
    print(og_data)
    print(f'\nNo of samples : {s_row}\nNo of features(dimensions) : {s_col}')

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Create a scatter plot
    plt.scatter(og_data['X'], og_data['Y'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.title('Scatter plot of original data : X and Y')
    plt.show()
    
    return og_data
    return s_row
    return s_col
    return cols

def sample4():

    """ stupid wall in one corner """

    # set the random seed for reproducibility
    np.random.seed(123)

    np.random.seed(123)

    # generate random points within an ellipse
    a = 2.0 # major axis
    b = 1.0 # minor axis
    theta = np.random.rand(100) * 2.0 * np.pi
    r = np.sqrt(np.random.rand(100))
    x = a * r * np.cos(theta)
    y = b * r * np.sin(theta)

    # generate 20 outlier points in one direction
    outliers_x = np.ones(20) * 4.0 # set x to a constant value of 4.0
    outliers_y = np.random.uniform(low=-1, high=1, size=20)

    # combine the x and y coordinates of the ellipse and outliers into a 2-column array
    X = np.vstack((np.column_stack((x, y)), np.column_stack((outliers_x, outliers_y))))
    og_data = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1]})
    s_row, s_col = og_data.shape
    cols = list(og_data.columns)

    # Print the DataFrame
    print(og_data)
    print(f'\nNo of samples : {s_row}\nNo of features(dimensions) : {s_col}')

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Create a scatter plot
    plt.scatter(og_data['X'], og_data['Y'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.title('Scatter plot of original data : X and Y')
    plt.show()
    
    return og_data
    return s_row
    return s_col
    return cols

def sample5():

    """ A fucking donuct thats supposed to be a shuriken """

    np.random.seed(42)

    n_samples = 100
    n_blades = 4

    theta = np.linspace(0, 2 * np.pi, n_samples)
    r_noise = 0.15 * np.random.randn(n_samples)
    r = 1 + r_noise

    x = np.array([])
    y = np.array([])

    for blade in range(n_blades):
        blade_theta = theta + blade * np.pi / n_blades
        blade_x = r * np.cos(blade_theta)
        blade_y = r * np.sin(blade_theta)

        # Make one blade unevenly longer
        if blade == 0:
            blade_x *= 1 + (0.5 * np.random.rand(n_samples))
            blade_y *= 1 + (0.5 * np.random.rand(n_samples))

        x = np.concatenate((x, blade_x))
        y = np.concatenate((y, blade_y))

    # Combine x and y as a dataset
    shuriken_data = np.column_stack((x, y))
    og_data = pd.DataFrame({'X': shuriken_data[:, 0], 'Y': shuriken_data[:, 1]})
    s_row, s_col = og_data.shape
    cols = list(og_data.columns)

    # Print the DataFrame
    print(og_data)
    print(f'\nNo of samples : {s_row}\nNo of features(dimensions) : {s_col}')

    # Set up Seaborn plotting style
    sns.set_style("whitegrid")

    # Plot the dataset using Seaborn
    sns.scatterplot(x=shuriken_data[:, 0], y=shuriken_data[:, 1])

    plt.axis("equal")
    plt.title("Uneven Shuriken-Shaped Dataset")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
    return og_data
    return s_row
    return s_col
    return cols



