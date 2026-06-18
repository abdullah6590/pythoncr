import os

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Linear Algebra: Interactive Masterclass</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <canvas id="particle-canvas"></canvas>

    <div class="container">

    <nav class="top-nav">
        <!-- Active item acts as the dropdown toggle on mobile -->
        <a href="#" class="nav-link active">📐 Linear Algebra</a>
        
        <!-- Other links to show when toggled -->
        <a href="basics.html" class="nav-link">🐍 Basics</a>
        <a href="index.html" class="nav-link">🚀 Functions</a>
        <a href="oop.html" class="nav-link">🏗️ OOP</a>
        <a href="numpy.html" class="nav-link">🧮 NumPy</a>
        <a href="pandas.html" class="nav-link">🐼 Pandas</a>
        <a href="matplotlib.html" class="nav-link">📊 Matplotlib</a>
    </nav>

    <header>
        <h1>📐 Linear Algebra Overview</h1>
        <p>Machine Learning experts cannot live without Linear Algebra. It is the branch of mathematics that concerns linear equations (and linear maps) and their representations in vector spaces and through matrices. Linear algebra is central to almost all areas of mathematics.</p>
    </header>

    <div class="storybook-intro">
        <h2>🤖 Essential for Machine Learning</h2>
        <p>The purpose of this chapter is to highlight the parts of linear algebra that are used in data science projects like machine learning and deep learning.</p>
        <ul>
            <li><span class="highlight-term">Scalars</span> : ML makes heavy use of Scalars</li>
            <li><span class="highlight-term">Vectors</span> : ML makes heavy use of Vectors</li>
            <li><span class="highlight-term">Matrices</span> : ML makes heavy use of Matrices</li>
            <li><span class="highlight-term">Tensors</span> : ML makes heavy use of Tensors</li>
        </ul>

        <h3 style="margin-top: 2rem;">📊 Visual Overview</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px; text-align: center;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <th style="padding: 8px; color: #38bdf8;">Scalar</th>
                <th style="padding: 8px; color: #38bdf8;">Vector(s)</th>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                <td style="padding: 10px;">1</td>
                <td style="padding: 10px;">[1]<br>[2]<br>[3]<br><br>[1 2 3]</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.2);">
                <th style="padding: 8px; color: #38bdf8; margin-top:10px;">Matrix</th>
                <th style="padding: 8px; color: #38bdf8; margin-top:10px;">Tensor</th>
            </tr>
            <tr>
                <td style="padding: 10px;">
                    [1 2 3]<br>
                    [4 5 6]
                </td>
                <td style="padding: 10px;">
                    [[1 2 3]<br>
                     [4 5 6]]<br>
                    <br>
                    [[4 5 6]<br>
                     [1 2 3]]
                </td>
            </tr>
        </table>
        
        <h3 style="margin-top: 2rem;">🗣️ The Language of Data</h3>
        <p><strong>Vectors and Matrices are the languages of data.</strong> With ML, most things are done with vectors and matrices. With vectors and matrices, you can Discover Secrets.</p>
    </div>

    <h2 class="module-title">Module 1: Deep Dive into Components</h2>

    <div class="task-card">
        <h3 class="task-title">Topic 1: Scalars</h3>
        <p class="task-desc">In linear algebra, a scalar is a single number. It has magnitude but no direction.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task1', this)">Show Answer</button>
        
        <div id="la-task1" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="comment"># In Python it can be written as a simple variable assignment:</span>
my_scalar = 1

x = 1
y = 1

print(my_scalar)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out1', this)">Show Output</button>
            <div id="out1" class="output-container">
1
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">my_scalar = 1</span> - We define a single number `1` and store it in a variable. This single numeric value is a scalar.</li>
                    <li><span class="code-snippet">x = 1</span> - Scalars are basic Python variables that hold primitive numbers like integers and floats.</li>
                    <li><strong>🤖 ML Connection:</strong> In Machine Learning, a scalar is typically a 0-dimensional tensor. It represents a single quantifiable value, like a learning rate, a loss value at a specific epoch, or a single weight before it's organized into a matrix.</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="storybook-intro">
        <h2>📌 Vectors Definition & Geometric Meaning</h2>
        <p>A vector is an ordered list of numbers. It mathematically represents a <strong>Point in space</strong>, and has both <strong>Direction</strong> and <strong>Magnitude</strong>.</p>
        
        <h3 style="margin-top: 1rem;">Types of Vectors:</h3>
        <ul>
            <li><strong>Row Vector:</strong> <code>[3 4]</code> — elements are arranged horizontally.</li>
            <li><strong>Column Vector:</strong> <code>[[3], [4]]</code> — elements are arranged vertically.</li>
        </ul>

        <h3 style="margin-top: 1rem;">Geometric Meaning:</h3>
        <p>For a vector <code>(3, 4)</code>, it means move <strong>3 units</strong> in the x direction and <strong>4 units</strong> in the y direction.</p>
        <p>The <strong>Magnitude</strong> (length) is calculated using the Pythagorean theorem: <code>√(3² + 4²) = √25 = 5</code>.</p>
        
        <h3 style="margin-top: 1rem;">🧠 In Machine Learning</h3>
        <p>A vector represents a data point, a feature set, or model weights. For example, if we have student data (Height, Weight, Age), one student is a vector: <code>x = [170, 65, 25]</code>.</p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 2: Vectors in Python</h3>
        <p class="task-desc">In Python, vectors are typically represented using a NumPy array.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task2', this)">Show Answer</button>
        
        <div id="la-task2" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

v = np.array([3, 4])
print("Vector:", v)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out2', this)">Show Output</button>
            <div id="out2" class="output-container">
Vector: [3 4]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">import numpy as np</span> - NumPy is the standard library for creating and manipulating arrays in Python.</li>
                    <li><span class="code-snippet">v = np.array([3, 4])</span> - We define a standard vector using a NumPy array.</li>
                </ul>
            </div>
        </div>
    </div>

    <h2 class="module-title">Module 2: Vector Operations</h2>

    <div class="task-card">
        <h3 class="task-title">Topic 3: Vector Arithmetic</h3>
        <p class="task-desc">Perform <strong>Addition</strong>, <strong>Subtraction</strong>, and <strong>Scalar Multiplication</strong> on vectors.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task3', this)">Show Answer</button>
        
        <div id="la-task3" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

a = np.array([1, 2])
b = np.array([3, 4])

<span class="comment"># Vector Addition</span>
print("Addition:", a + b)

<span class="comment"># Vector Subtraction</span>
print("Subtraction:", a - b)

<span class="comment"># Scalar Multiplication</span>
print("Scalar Multiplication:", 2 * a)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out3', this)">Show Output</button>
            <div id="out3" class="output-container">
Addition: [4 6]<br>
Subtraction: [-2 -2]<br>
Scalar Multiplication: [2 4]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">a + b</span> - Geometric meaning: Head-to-tail addition. `[1+3, 2+4] = [4, 6]`.</li>
                    <li><span class="code-snippet">a - b</span> - Subtracts elements one by one. `[1-3, 2-4] = [-2, -2]`.</li>
                    <li><span class="code-snippet">2 * a</span> - Geometric meaning: Stretching or shrinking the vector. `[2*1, 2*2] = [2, 4]`.</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 4: Dot Product (VERY IMPORTANT)</h3>
        <p class="task-desc">The Dot Product mathematically sums the products of the corresponding entries of two sequences of numbers. <code>a · b = a₁b₁ + a₂b₂</code>.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task4', this)">Show Answer</button>
        
        <div id="la-task4" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

a = np.array([1, 2])
b = np.array([3, 4])

<span class="comment"># Element-wise: (1 x 3) + (2 x 4) = 3 + 8 = 11</span>
print("Dot Product:", np.dot(a, b))</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out4', this)">Show Output</button>
            <div id="out4" class="output-container">
Dot Product: 11
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.dot(a, b)</span> - NumPy's built-in function to compute the dot product of two arrays.</li>
                    <li><strong>Geometric Meaning:</strong> <code>a · b = |a||b| cos(θ)</code>. If the dot product is 0, the vectors are perpendicular!</li>
                    <li><strong>🤖 ML Connection:</strong> Essential in Linear Regression, Neural Networks, Attention mechanisms, and Similarity measurement (Cosine Similarity). Standard example: <code>prediction = w^T * x</code>.</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="storybook-intro">
        <h2>✖️ Cross Product (Basic Idea)</h2>
        <p>The cross product is only defined in 3D: <code>a = (a₁, a₂, a₃)</code> and <code>b = (b₁, b₂, b₃)</code>.</p>
        <p>The result is a new <strong>vector that is perpendicular to both</strong> <code>a</code> and <code>b</code>.</p>
        <p><em>Note: It is not heavily used in basic Machine Learning algorithms.</em></p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 5: Vector Norms (Magnitude)</h3>
        <p class="task-desc">A norm is a mathematical way to define the "Length" or "Magnitude" of a vector.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task5', this)">Show Answer</button>
        
        <div id="la-task5" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

v = np.array([3, 4])

<span class="comment"># L2 Norm (Euclidean norm)</span>
print("L2 Norm:", np.linalg.norm(v))

<span class="comment"># L1 Norm (Manhattan norm)</span>
print("L1 Norm:", np.linalg.norm(v, 1))</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out5', this)">Show Output</button>
            <div id="out5" class="output-container">
L2 Norm: 5.0<br>
L1 Norm: 7.0
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.norm(v)</span> - Computes the <strong>L2 Norm (Euclidean norm)</strong>. Formula: <code>√(v₁² + v₂²)</code>. Example: <code>√(3² + 4²) = 5</code>.</li>
                    <li><span class="code-snippet">np.linalg.norm(v, 1)</span> - Computes the <strong>L1 Norm</strong>. Formula: <code>|v₁| + |v₂|</code>. Example: <code>|3| + |4| = 7</code>.</li>
                    <li><strong>🤖 ML Connection:</strong> Norms are used in <strong>Regularization</strong>. L2 is used in Ridge regression, while L1 is used in Lasso regression to prevent overfitting.</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 6: Distance Between Vectors</h3>
        <p class="task-desc">Finding the Euclidean distance between two points in space.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task6', this)">Show Answer</button>
        
        <div id="la-task6" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

a = np.array([1, 2])
b = np.array([4, 6])

<span class="comment"># Distance = ||a - b||</span>
distance = np.linalg.norm(a - b)
print("Distance:", distance)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out6', this)">Show Output</button>
            <div id="out6" class="output-container">
Distance: 5.0
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.norm(a - b)</span> - Formula is <code>√((4 - 1)² + (6 - 2)²) = √(3² + 4²) = 5</code>. First we subtract the vectors, then compute the L2 norm of the resulting difference vector!</li>
                    <li><strong>🤖 ML Connection:</strong> Heavily used in <strong>KNN (K-Nearest Neighbors)</strong>, <strong>Clustering (e.g., K-Means)</strong>, and measuring <strong>Similarity</strong> between embeddings.</li>
                </ul>
            </div>
        </div>
    </div>


    <h2 class="module-title">Module 3: Matrices Overview</h2>

    <div class="storybook-intro">
        <h2>📌 What is a Matrix?</h2>
        <p>A matrix is a rectangular arrangement of numbers into rows and columns.</p>
        
        <h3 style="margin-top: 1rem;">Matrix Size / Shape:</h3>
        <p>If a matrix has <code>m</code> rows and <code>n</code> columns, we say it is an <code>m × n</code> (m by n) matrix.</p>
        <p>Example of a 3 × 2 matrix:</p>
        <pre style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; color:#38bdf8;">
[ 1  2 ]
[ 3  4 ]
[ 5  6 ]</pre>

        <h3 style="margin-top: 1rem;">🤖 In Machine Learning</h3>
        <p><strong>A dataset is always a matrix!</strong> The rows represent samples (instances) and columns represent features (variables).</p>
        <p>Example: <code>X</code> where Rows = 2 students, Columns = (Height, Weight, Age).</p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 7: Matrices in Python (Shape)</h3>
        <p class="task-desc">Creating a data matrix and checking its shape using NumPy.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task7', this)">Show Answer</button>
        
        <div id="la-task7" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

<span class="comment"># Rows: Samples, Cols: Features (Height, Weight, Age)</span>
X = np.array([
    [170, 65, 25],
    [180, 75, 30]
])

print("Shape:", X.shape)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out7', this)">Show Output</button>
            <div id="out7" class="output-container">
Shape: (2, 3)
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">X = np.array([[...], [...]])</span> - Creates a 2D grouping of numbers. This is our matrix `X`.</li>
                    <li><span class="code-snippet">X.shape</span> - NumPy uses the `.shape` attribute to return a tuple `(rows, columns)`. Here we see 2 rows and 3 columns!</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="storybook-intro">
        <h2>🏷️ Types of Matrices</h2>
        <ul>
            <li><strong>Row Matrix:</strong> Only 1 row. e.g. <code>[1 2 3]</code></li>
            <li><strong>Column Matrix:</strong> Only 1 column.</li>
            <li><strong>Square Matrix:</strong> Same number of rows and columns (e.g. 2×2, 3×3).</li>
            <li><strong>Zero Matrix:</strong> All elements are zero.</li>
            <li><strong>Diagonal Matrix:</strong> Only diagonal elements are non-zero. e.g. <code>[[3, 0], [0, 5]]</code></li>
            <li><strong>Identity Matrix (Very Important):</strong> Diagonal elements are 1, everything else is 0. Denoted as <code>I</code>. Property: <code>AI = A</code>.</li>
        </ul>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 8: Creating Types of Matrices</h3>
        <p class="task-desc">In Python, we can generate Identity matrices quickly.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task8', this)">Show Answer</button>
        
        <div id="la-task8" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

<span class="comment"># Create a 3x3 Identity Matrix</span>
I = np.eye(3)

print("Identity Matrix:")
print(I)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out8', this)">Show Output</button>
            <div id="out8" class="output-container">
Identity Matrix:<br>
[[1. 0. 0.]<br>
 [0. 1. 0.]<br>
 [0. 0. 1.]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.eye(3)</span> - `eye` is a clever naming convention in NumPy (sounds like 'I' for Identity). It returns a 2-D array with ones on the diagonal and zeros elsewhere.</li>
                </ul>
            </div>
        </div>
    </div>

    <h2 class="module-title">Module 4: Matrix Operations</h2>

    <div class="task-card">
        <h3 class="task-title">Topic 9: Matrix Addition and Scalar Mul</h3>
        <p class="task-desc">Addition is done element-wise, <strong>only if dimensions are the same</strong>. Scalar multiplication scales every individual element.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task9', this)">Show Answer</button>
        
        <div id="la-task9" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

<span class="comment"># Addition</span>
print("A + B =\n", A + B)

<span class="comment"># Scalar Multiplication</span>
print("\n2 * A =\n", 2 * A)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out9', this)">Show Output</button>
            <div id="out9" class="output-container">
A + B =<br>
[[ 6  8]<br>
 [10 12]]<br>
<br>
2 * A =<br>
[[ 2  4]<br>
 [ 6  8]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">A + B</span> - At row 0 col 0: 1 + 5 = 6. At row 0 col 1: 2 + 6 = 8. And so on.</li>
                    <li><span class="code-snippet">2 * A</span> - Every element is multiplied by 2: 1->2, 2->4, 3->6, 4->8.</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="task-card">
        <h3 class="task-title">Topic 10: Matrix Multiplication (VERY IMPORTANT)</h3>
        <p class="task-desc">This is the <strong>core of neural networks</strong>. To multiply <code>A * B</code>, the inner dimensions must match: <code>A(m×n) * B(n×p) = C(m×p)</code>.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task10', this)">Show Answer</button>
        
        <div id="la-task10" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[1, 2], [3, 4]])  <span class="comment"># 2x2</span>
B = np.array([[5], [6]])        <span class="comment"># 2x1</span>

<span class="comment"># Using the @ operator for matrix multiplication</span>
print("A @ B =\n", A @ B)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out10', this)">Show Output</button>
            <div id="out10" class="output-container">
A @ B =<br>
[[17]<br>
 [39]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">A @ B</span> - The `@` operator in Python (or `np.matmul()`) performs matrix multiplication.</li>
                    <li><strong>Math Behind It:</strong> Row 1 of A * Col 1 of B: (1*5) + (2*6) = 5 + 12 = 17. Row 2 of A * Col 1 of B: (3*5) + (4*6) = 15 + 24 = 39.</li>
                    <li><strong>🤖 ML Connection:</strong> In Neural Networks, a layer performs a Forward Propagation operation: <code>Z = X @ W + b</code> (Inputs * Weights Matrix + Bias). Deep learning is essentially repeated matrix multiplication!</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="storybook-intro">
        <h2>🔥 Why Matrices Matter in ML</h2>
        <ul>
            <li><strong>Data = Matrix:</strong> Your entire dataset is mathematically a matrix.</li>
            <li><strong>Weights = Matrix:</strong> Neural network connections between layers are matrices.</li>
            <li><strong>Operations = Matrix Multiplication:</strong> Forward passes are just large matrix multiplications.</li>
            <li><strong>PCA = Covariance Matrix:</strong> Principal Component Analysis uses eigenvectors of a covariance matrix.</li>
            <li><strong>Optimization = Hessian Matrix:</strong> Advanced optimization uses the Hessian matrix of second derivatives.</li>
        </ul>
    </div>


    <h2 class="module-title">Module 5: Matrix Properties & Transformations</h2>

    <div class="task-card">
        <h3 class="task-title">Topic 11: Transpose</h3>
        <p class="task-desc">To transpose a matrix, you swap its rows and columns. Written as <code>Aᵀ</code>.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task11', this)">Show Answer</button>
        
        <div id="la-task11" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([
    [1, 2],
    [3, 4]
])

print("Transposed:\n", A.T)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out11', this)">Show Output</button>
            <div id="out11" class="output-container">
Transposed:<br>
[[1 3]<br>
 [2 4]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">A.T</span> - The `.T` attribute in NumPy instantly returns the transpose. What was row 1 (`1, 2`) becomes column 1!</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 12: Determinant</h3>
        <p class="task-desc">Only for square matrices. If the determinant equals 0, the matrix is <strong>not invertible</strong> (singular).</p>
        <button class="btn-toggle" onclick="toggleCode('la-task12', this)">Show Answer</button>
        
        <div id="la-task12" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[1, 2], [3, 4]])

<span class="comment"># det = (ad - bc) = (1*4 - 2*3) = 4 - 6 = -2</span>
print("Determinant:", np.linalg.det(A))</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out12', this)">Show Output</button>
            <div id="out12" class="output-container">
Determinant: -2.0000000000000004
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.det(A)</span> - Uses NumPy's linear algebra (`linalg`) module to compute the determinant. The tiny float rounding error is standard in binary floating-point computation.</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 13: Inverse of a Matrix</h3>
        <p class="task-desc">If <code>A * A⁻¹ = I</code> (Identity Matrix), then <code>A⁻¹</code> is the inverse.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task13', this)">Show Answer</button>
        
        <div id="la-task13" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[1, 2], [3, 4]])

print("Inverse of A:\n", np.linalg.inv(A))</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out13', this)">Show Output</button>
            <div id="out13" class="output-container">
Inverse of A:<br>
[[-2.   1. ]<br>
 [ 1.5 -0.5]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.inv(A)</span> - Computes the multiplicative inverse of a matrix.</li>
                    <li><strong>🤖 ML Connection:</strong> Finding the exact "closed-form" solution to Linear Regression relies heavily on the inverse: `w = (XᵀX)⁻¹ Xᵀy`. For large datasets, taking the inverse is computationally expensive so we use Gradient Descent instead!</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 14: Rank of Matrix</h3>
        <p class="task-desc">The Rank is the number of linearly independent rows or columns in the matrix.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task14', this)">Show Answer</button>
        
        <div id="la-task14" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[1, 2], [3, 4]])

print("Rank:", np.linalg.matrix_rank(A))</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out14', this)">Show Output</button>
            <div id="out14" class="output-container">
Rank: 2
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.matrix_rank(A)</span> - Calculates rank using singular value decomposition (SVD) under the hood.</li>
                    <li><strong>🤖 ML Connection:</strong> If the rank of your feature matrix `X` is less than the number of columns, you have a **multicollinearity problem** (some features are perfectly correlated and redundant).</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="task-card">
        <h3 class="task-title">Topic 15: Tensors</h3>
        <p class="task-desc">A Tensor is an N-dimensional Matrix. While a scalar is 0D, vector is 1D, and matrix is 2D, a tensor can be 3D, 4D, or any number of dimensions.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task15', this)">Show Answer</button>
        
        <div id="la-task15" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="comment"># In Python, a tensor is a NumPy array with multiple dimensions,</span>
<span class="comment"># nested deeper than 2 levels. Here's a 3D Tensor:</span>
<span class="keyword">import</span> numpy <span class="keyword">as</span> np

my_tensor = np.array([
  [ <span class="comment"># Block 1</span>
    [1, 2, 3],
    [4, 5, 6]
  ],
  [ <span class="comment"># Block 2</span>
    [4, 5, 6],
    [1, 2, 3]
  ]
])

print(my_tensor[0, 1, 2]) <span class="comment"># Accessing an element deeply nested</span></code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out15', this)">Show Output</button>
            <div id="out15" class="output-container">
6
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">my_tensor = np.array([...])</span> - We create an array containing two 2D matrices. This makes it a 3D tensor with a shape of `(2, 2, 3)` (2 blocks, 2 rows, 3 columns).</li>
                    <li><span class="code-snippet">my_tensor[0, 1, 2]</span> - Accessing the tensor requires 3 indices! Block 0, Row 1, Column 2. Let's trace it: Block 0 is the first matrix. Row 1 is `[4, 5, 6]`. Column 2 is the 3rd item: `6`!</li>
                    <li><strong>🤖 ML Connection:</strong> Deep Learning frameworks like TensorFlow and PyTorch are named after Tensors! Images are 3D tensors (Height x Width x Color Channels). Video data is a 4D tensor (Frames x Height x Width x Colors). Tensors elegantly hold exceptionally complex structured data.</li>
                </ul>
            </div>
        </div>
    </div>


    <h2 class="module-title">Module 6: Systems of Linear Equations</h2>

    <div class="storybook-intro">
        <h2>📌 What is a System of Linear Equations?</h2>
        <p>A system of equations looks like this:</p>
        <pre style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; color:#38bdf8;">
2x + y = 5
 x - y = 1</pre>
        <p>We want to find: <code>x = ?</code>, <code>y = ?</code></p>

        <h3 style="margin-top: 1rem;">🧠 Geometric Meaning</h3>
        <p>Each equation represents a <strong>line</strong>. The solution is the intersection point of the lines.</p>
        <ul>
            <li><strong>Lines intersect:</strong> One unique solution.</li>
            <li><strong>Lines are parallel:</strong> No solution.</li>
            <li><strong>Same line:</strong> Infinite solutions.</li>
        </ul>
    </div>

    <div class="storybook-intro">
        <h2>🔢 Matrix Form of Linear System</h2>
        <p>The system from above can be written as <strong><code>AX = B</code></strong>:</p>
        <pre style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; color:#38bdf8;">
A = [ 2  1 ]      X = [ x ]      B = [ 5 ]
    [ 1 -1 ]          [ y ]          [ 1 ]</pre>
        
        <h3 style="margin-top: 1rem;">🔥 Why Important in ML?</h3>
        <p><strong>Linear Regression</strong> is mathematically equivalent to: <code>Xw = y</code> (where we solve for weights `w`). Neural networks also solve huge systems of equations internally!</p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 16: Solving Using Inverse</h3>
        <p class="task-desc">If the matrix <code>A</code> is invertible, we can solve for <code>X</code> algebraically: <code>X = A⁻¹ B</code>.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task16', this)">Show Answer</button>
        
        <div id="la-task16" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[2, 1], [1, -1]])
B = np.array([5, 1])

<span class="comment"># X = A_inverse @ B</span>
X = np.linalg.inv(A) @ B

print("Solution [x, y]:", X)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out16', this)">Show Output</button>
            <div id="out16" class="output-container">
Solution [x, y]: [2. 1.]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.inv(A) @ B</span> - Computes the inverse of A, then matrix-multiplies it with B. The solution perfectly finds x = 2 and y = 1.</li>
                    <li><strong>⚠️ Real World ML:</strong> While mathematically correct, using the inverse directly is memory-heavy and slow for huge matrices! Instead, computational algorithms prefer elimination or decomposition.</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="storybook-intro">
        <h2>🛠️ Gaussian Elimination & Row Echelon Form</h2>
        <p><strong>Gaussian Elimination</strong> converts a system (an augmented matrix) into an <strong>upper triangular form</strong> by performing basic row operations. Once triangular, the system is solved easily by back substitution.</p>
        
        <h3 style="margin-top: 1rem;">Row Echelon Form (REF):</h3>
        <ul>
            <li>All zeros below the leading coefficient (pivot).</li>
            <li>Pivots move rightwards as you go downwards.</li>
        </ul>
        <pre style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; color:#38bdf8;">
[ 1  2  3 ]
[ 0  1  4 ]
[ 0  0  2 ]</pre>

        <h3 style="margin-top: 1rem;">Reduced Row Echelon Form (RREF):</h3>
        <p>A stricter version: Every pivot must be exactly <strong>1</strong>, and there are <strong>zeros above AND below</strong> every pivot.</p>
        <pre style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; color:#38bdf8;">
[ 1  0  2 ]
[ 0  1  3 ]
[ 0  0  0 ]</pre>
        <p><strong>🧠 Why Important?</strong> Gaussian elimination forms the foundation of modern linear solvers, LU decomposition, and deep learning backpropagation math.</p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 17: Reduced Row Echelon Form with SymPy</h3>
        <p class="task-desc">In Python, we can use the <code>SymPy</code> library to easily find the exact RREF of any matrix without floating-point errors.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task17', this)">Show Answer</button>
        
        <div id="la-task17" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">from</span> sympy <span class="keyword">import</span> Matrix

<span class="comment"># Define the augmented matrix [A | B]</span>
A = Matrix([
    [2,  1, 5],
    [1, -1, 1]
])

rref_matrix, pivot_columns = A.rref()

print("RREF of A:\n", rref_matrix)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out17', this)">Show Output</button>
            <div id="out17" class="output-container">
RREF of A:<br>
Matrix([[1, 0, 2], [0, 1, 1]])
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">Matrix([...])</span> - SymPy uses exact symbolic calculation. It avoids the rounding errors you often see with standard floating-point representation!</li>
                    <li><span class="code-snippet">A.rref()</span> - Calculates the Reduced Row Echelon Form. The output matrix tells us perfectly: x=2, y=1.</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="task-card">
        <h3 class="task-title">Topic 18: LU Decomposition</h3>
        <p class="task-desc">Instead of solving directly, we decompose the original matrix into two pieces: <code>A = LU</code> (Lower and Upper triangular matrices).</p>
        <button class="btn-toggle" onclick="toggleCode('la-task18', this)">Show Answer</button>
        
        <div id="la-task18" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np
<span class="keyword">import</span> scipy.linalg <span class="keyword">as</span> la

A = np.array([[2, 1], [1, -1]])

<span class="comment"># Decompose A into P, L, U</span>
P, L, U = la.lu(A)

print("L (Lower):\n", L)
print("\nU (Upper):\n", U)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out18', this)">Show Output</button>
            <div id="out18" class="output-container">
L (Lower):<br>
[[1.  0. ]<br>
 [0.5 1. ]]<br>
<br>
U (Upper):<br>
[[ 2.   1. ]<br>
 [ 0.  -1.5]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">scipy.linalg.lu(A)</span> - SciPy is designed for deep scientific computing. `la.lu` decomposes `A` structurally and returns a Permutation matrix (P), Lower triangular matrix (L), and Upper triangular matrix (U).</li>
                    <li><strong>🧠 Why Important in ML?</strong> LU decomposition is heavily used in super-fast linear solvers, advanced modeling, scientific simulations, and <strong>Model Predictive Control (MPC) problems!</strong></li>
                </ul>
            </div>
        </div>
    </div>

    <div class="storybook-intro">
        <h2>🔥 Final ML System Connections</h2>
        
        <h3 style="margin-top: 1rem;">Finding Solutions Based on Rank:</h3>
        <ul>
            <li><code>rank(A) = rank([A|B]) = number of variables</code> → <strong>Unique solution</strong></li>
            <li><code>rank(A) < rank([A|B])</code> → <strong>No solution</strong></li>
            <li><code>rank(A) = rank([A|B]) < number of variables</code> → <strong>Infinite solutions</strong></li>
        </ul>

        <h3 style="margin-top: 1rem;">🤖 How it Connects to Everything:</h3>
        <p>This entire section provides the mathematical framework for:</p>
        <ul>
            <li>✔ <strong>Linear Regression & Least Squares</strong> (Fitting curves to data points)</li>
            <li>✔ <strong>Optimization Engines</strong> (Guiding gradient descent trajectories)</li>
            <li>✔ <strong>Control Systems & Model Predictive Control (MPC)</strong> (Simulating and controlling robots or self-driving constraints over continuous loops!)</li>
        </ul>
    </div>


    <h2 class="module-title">Module 7: Vector Spaces (Foundation of ML Theory)</h2>

    <div class="storybook-intro">
        <h2>📌 Vector Space & Subspace</h2>
        
        <h3 style="margin-top: 1rem;">What is a Vector Space?</h3>
        <p>A vector space is a collection of vectors where you can <strong>add vectors</strong> and <strong>multiply by scalars</strong>, and the mathematical result stays inside the same overall space.</p>
        <p><strong>Example:</strong> All 2D vectors mathematically form the space ℝ² (R-squared). <code>{"[x, y]"}</code>.</p>
        
        <h3 style="margin-top: 1rem;">🧠 ML Connection</h3>
        <p>Suppose your dataset has two features: Height and Weight. Each data point is <code>x = [height, weight]</code>. All possible such vectors make up ℝ². Therefore, your <strong>Feature Space = Vector Space</strong>.</p>

        <h3 style="margin-top: 1rem;">What is a Subspace?</h3>
        <p>A subspace is literally a smaller space perfectly nested inside a larger vector space.</p>
        <p>For example, all vectors of the form <code>[x, 0]</code> form a horizontal line, which is a subspace of the full 2D plane ℝ².</p>
        
        <h3 style="margin-top: 1rem;">🤖 ML Example (Subspace Projection)</h3>
        <p>Suppose you decide to completely remove the "weight" feature. Now your data point is <code>x = [height, 0]</code>. You just reduced the dimension! This feature reduction is mathematically a <strong>subspace projection</strong>.</p>
    </div>

    <div class="storybook-intro">
        <h2>➕ Linear Combination & Span</h2>
        
        <h3 style="margin-top: 1rem;">Linear Combination</h3>
        <p>If you have vectors <code>v₁</code> and <code>v₂</code>, then <code>a·v₁ + b·v₂</code> (where <em>a</em> and <em>b</em> are numbers) is a linear combination!</p>
        
        <h3 style="margin-top: 1rem;">🧠 ML Connection</h3>
        <p>In <strong>Linear Regression</strong>, you do exactly this! <code>y = w₁·x₁ + w₂·x₂</code>. If `x₁` is height and `x₂` is weight, your final prediction is a linear combination of those features!</p>

        <h3 style="margin-top: 1rem;">Span</h3>
        <p>The <strong>span</strong> of a set of vectors is all possible linear combinations of them. If you have <code>v₁=[1, 0]</code> and <code>v₂=[0, 1]</code>, their span is the entire 2D space ℝ²! You can reach any point mathematically.</p>

        <h3 style="margin-top: 1rem;">🤖 ML Example (Multicollinearity)</h3>
        <p>If your dataset has two independent features (Height, Weight), they span a sturdy 2D feature space.</p>
        <p>But wait... what if one feature is completely dependent on the other? What if <code>Weight = 2 × Height</code>? Then their span structurally collapses into just a 1D straight line! This reduced dimension causes the dreaded <strong>multicollinearity problem</strong> in ML.</p>
    </div>

    <div class="storybook-intro">
        <h2>📐 Linear Independence, Basis & Dimension</h2>
        
        <h3 style="margin-top: 1rem;">Linear Independence</h3>
        <p>Vectors are linearly independent if no vector in the set can be written as a linear combination of the others.</p>
        
        <h3 style="margin-top: 1rem;">🤖 ML Connection</h3>
        <p>If Feature 1 = Height, and Feature 2 = (2 × Height), they are linearly dependent. In ML, this makes the matrix `(XᵀX)` impossible to invert (it becomes singular). The mathematical math breaks down, causing massive model instability!</p>

        <h3 style="margin-top: 1rem;">Basis</h3>
        <p>A <strong>Basis</strong> is a minimal set of independent vectors that spans the whole space. For ℝ², <code>[1, 0]</code> and <code>[0, 1]</code> form a basis.</p>

        <h3 style="margin-top: 1rem;">🤖 ML Example (PCA)</h3>
        <p>Suppose your original features are Height, Weight, and Age. <strong>Principal Component Analysis (PCA)</strong> crunches these into new synthetic vectors called PC1 and PC2. These new Principal Components mathematically become the <strong>new basis</strong> for a smaller subspace. That is the exact mechanism of dimensionality reduction!</p>

        <h3 style="margin-top: 1rem;">Dimension</h3>
        <p>Dimension is simply the number of basis vectors. ℝ² = dimension 2. If your dataset has 100 features, its dimension is 100. If PCA knocks it down to 10 features, the dimension is reduced to 10, drastically lowering noise and compute time.</p>
    </div>



    <h2 class="module-title">Module 8: Eigenvalues & Vectors (Heart of ML)</h2>

    <div class="storybook-intro">
        <h2>📌 1. What is an Eigenvector?</h2>
        
        <p>Matrix multiplication normally changes the direction of a vector. But an <strong>eigenvector</strong> is special:</p>
        <ul>
            <li>👉 It keeps the <strong>same direction</strong></li>
            <li>👉 It only gets <strong>scaled</strong> (stretched or shrunk)</li>
        </ul>

        <h3 style="margin-top: 1rem;">Mathematical Definition</h3>
        <p>For a square matrix `A`, if there exists a vector `v ≠ 0` such that:</p>
        <pre style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; color:#38bdf8;">A v = λ v</pre>
        <p>Then <strong><code>v</code> is the eigenvector</strong>, and <strong><code>λ</code> is the eigenvalue</strong>.</p>
        
        <h3 style="margin-top: 1rem;">🤖 ML Intuition</h3>
        <p>Imagine your data is a cloud of points stretched in some direction. <strong>Eigenvectors show the main directions of the data spread</strong>. They highlight the most important patterns mathematically! This is exactly what Principal Component Analysis (PCA) finds.</p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 19: Finding Eigenvalues & Eigenvectors</h3>
        <p class="task-desc">Solving <code>det(A - λI) = 0</code> manually gives the Characteristic Equation. In Python, NumPy does it instantly.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task19', this)">Show Answer</button>
        
        <div id="la-task19" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[2, 1],
              [1, 2]])

<span class="comment"># Calculate eigenvalues and eigenvectors</span>
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out19', this)">Show Output</button>
            <div id="out19" class="output-container">
Eigenvalues: [3. 1.]<br>
Eigenvectors:<br>
 [[ 0.70710678 -0.70710678]<br>
  [ 0.70710678  0.70710678]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.eig(A)</span> - The NumPy function that returns a tuple of (eigenvalues, eigenvectors).</li>
                    <li><strong>Geometric Meaning:</strong> By taking the vectors, the eigenvalue `3.` tells us there is strong stretching in that specific direction. The corresponding eigenvector `[0.707, 0.707]` indicates a 45-degree diagonal. The `1.` eigenvalue implies no stretching, just preserving length in the orthogonal direction `[-0.707, 0.707]`.</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="storybook-intro">
        <h2>🛠️ Advanced Matrix Properties</h2>

        <h3 style="margin-top: 1rem;">1. Diagonalization (`A = PDP⁻¹`)</h3>
        <p>If a matrix has enough independent eigenvectors, it can be decomposed into <code>P</code> (matrix of eigenvectors) and <code>D</code> (diagonal matrix of eigenvalues). This massively simplifies computing matrix powers and studying system stability (Crucial for <strong>Markov models</strong> and <strong>Control systems / MPC</strong>).</p>

        <h3 style="margin-top: 1rem;">2. The Spectral Theorem</h3>
        <p>If a matrix is <strong>Symmetric</strong>, all its eigenvalues are real numbers, and its eigenvectors are perfectly orthogonal (perpendicular). Covariance matrices are always symmetric, which is why PCA works so nicely and cleanly.</p>
        
        <h3 style="margin-top: 1rem;">3. Positive Definite Matrices</h3>
        <p>If all eigenvalues are <code>> 0</code>, the matrix is <em>Positive Definite</em>. In optimization (like computing the Hessian matrix for Neural Networks), this guarantees you have found a global convex minimum! In linear regression `(XᵀX)`, positive eigenvalues guarantee a stable, unique model with no multicollinearity!</p>
    </div>


    <div class="task-card">
        <h3 class="task-title">Topic 20: PCA From Scratch (Deep Learning Connection)</h3>
        <p class="task-desc">Principal Component Analysis is nothing more than taking the Eigen Decomposition of a Covariance Matrix.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task20', this)">Show Answer</button>
        
        <div id="la-task20" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

<span class="comment"># Sample student data: [Height, Weight]</span>
X = np.array([[170, 65],
              [180, 75],
              [175, 70]])

<span class="comment"># Step 1: Center data by subtracting the mean</span>
X_mean = X - np.mean(X, axis=0)

<span class="comment"># Step 2: Compute Covariance matrix (Features are rows, so we transpose)</span>
C = np.cov(X_mean.T)

<span class="comment"># Step 3: Eigen decomposition!</span>
eigenvalues, eigenvectors = np.linalg.eig(C)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out20', this)">Show Output</button>
            <div id="out20" class="output-container">
Eigenvalues: [50.  0.]<br>
Eigenvectors:<br>
 [[ 0.70710678 -0.70710678]<br>
  [ 0.70710678  0.70710678]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">X_mean = X - np.mean(...)</span> - We must center data at coordinate (0,0) before analyzing variance.</li>
                    <li><span class="code-snippet">C = np.cov(X_mean.T)</span> - The Covariance matrix measures how Height and Weight change together. It is symmetric!</li>
                    <li><span class="code-snippet">np.linalg.eig(C)</span> - We find the eigenvectors. The largest eigenvalue is `50.`! Therefore, the corresponding eigenvector `[0.707, 0.707]` IS the **First Principal Component**. </li>
                    <li><strong>🤖 Neural Networks Connection:</strong> In deep learning, the eigenvalues of the weight matrices dictate gradient explosion (eigenvalues > 1) and gradient vanishing (eigenvalues < 1), dictating the entire training stability!</li>
                </ul>
            </div>
        </div>
    </div>


    <h2 class="module-title">Module 9: Orthogonality & Gram–Schmidt</h2>

    <div class="storybook-intro">
        <h2>📌 1. Orthogonal Vectors</h2>
        
        <p>Two vectors are strictly <strong>orthogonal</strong> if their dot product equals exactly zero: <code>v₁ · v₂ = 0</code>. Geometrically, this means they are perfectly perpendicular (at a 90-degree angle).</p>
        
        <h3 style="margin-top: 1rem;">🧠 Geometric Meaning</h3>
        <p>Orthogonal vectors carry completely independent information. They do not overlap in direction at all.</p>
        
        <h3 style="margin-top: 1rem;">🤖 ML Connection: PCA & Regression</h3>
        <p>In PCA, principal components are calculated to be entirely orthogonal. Why? Because we want each new component to capture uniquely new mathematical information without redundancy. If your features are orthogonal, you have <strong>zero multicollinearity</strong> and perfectly stable regression.</p>
    </div>

    <div class="storybook-intro">
        <h2>📌 2. Orthonormal Vectors</h2>
        <p>Vectors that are Orthogonal <strong>AND</strong> have a length (magnitude) perfectly equal to 1. <code>||v|| = 1</code>.</p>
        
        <h3 style="margin-top: 1rem;">🤖 ML Connection: Neural Networks</h3>
        <p>In deep neural networks, weight matrix initialization sometimes algorithmically forces matrices to be orthonormal. This elegantly prevents <strong>gradient explosion</strong> and maintains perfect variance stability across 100+ deep layers.</p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 21: Orthogonal Projections</h3>
        <p class="task-desc">Projection gives the "shadow" of one vector onto another. This mathematically minimizes the perpendicular error distance.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task21', this)">Show Answer</button>
        
        <div id="la-task21" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

x = np.array([1, 2])
y = np.array([3, 4])

<span class="comment"># Projection of y onto x</span>
<span class="comment"># Formula: proj_x(y) = ((y · x) / (x · x)) * x</span>
proj = (np.dot(y, x) / np.dot(x, x)) * x

print("Projection:", proj)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out21', this)">Show Output</button>
            <div id="out21" class="output-container">
Projection: [2.2 4.4]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">(np.dot(y, x) / np.dot(x, x)) * x</span> - First computes the scalar mapping ratio (dot product), divides by the length of `x` squared, and then multiplies that scalar back against vector `x`.</li>
                    <li><strong>🤖 ML Connection:</strong> Linear regression finds the mathematical projection of the Target vector <code>y</code> directly onto the Column space of <code>X</code>. Because projection perfectly minimizes distance, it minimizes error. That is why Least Squares Regression works conceptually!</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="task-card">
        <h3 class="task-title">Topic 22: The Gram–Schmidt Process</h3>
        <p class="task-desc">This process algorithmically converts a set of standard dependent/independent vectors into a perfectly orthogonal set.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task22', this)">Show Answer</button>
        
        <div id="la-task22" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

v1 = np.array([1, 1])
v2 = np.array([1, 0])

<span class="comment"># Step 1: Base vector</span>
u1 = v1

<span class="comment"># Step 2: Remove the projection of v2 onto u1 from v2!</span>
proj = (np.dot(v2, u1) / np.dot(u1, u1)) * u1
u2 = v2 - proj

print("Original:\nv1:\n", v1, "\nv2:\n", v2)
print("\nOrthogonal vectors:")
print("u1:\n", u1)
print("u2:\n", u2)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out22', this)">Show Output</button>
            <div id="out22" class="output-container">
Original:<br>
v1:<br>
 [1 1]<br>
v2:<br>
 [1 0]<br>
<br>
Orthogonal vectors:<br>
u1:<br>
 [1 1]<br>
u2:<br>
 [ 0.5 -0.5]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">u2 = v2 - proj</span> - The core magic. We take `v2` and cleanly subtract the "shadow" it casts onto `u1`. The remaining vector `u2` is mathematically guaranteed to be orthogonal to `u1`! (Notice the dot product of `[1, 1]` and `[0.5, -0.5]` is zero).</li>
                    <li><strong>🧠 Why Important?</strong> Gram-Schmidt is historically foundational to creating QR Decomposition algorithms!</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="task-card">
        <h3 class="task-title">Topic 23: QR Decomposition</h3>
        <p class="task-desc">Factorizes any matrix <code>A</code> into <code>A = QR</code>, where Q is orthogonal/orthonormal and R is an upper triangular matrix.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task23', this)">Show Answer</button>
        
        <div id="la-task23" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

A = np.array([[1, 1],
              [1, 0]])

<span class="comment"># Decompose into Q (orthonormal) and R (upper triangular)</span>
Q, R = np.linalg.qr(A)

print("Q:\n", Q)
print("\nR:\n", R)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out23', this)">Show Output</button>
            <div id="out23" class="output-container">
Q:<br>
 [[-0.70710678 -0.70710678]<br>
  [-0.70710678  0.70710678]]<br>
<br>
R:<br>
 [[-1.41421356 -0.70710678]<br>
  [ 0.         -0.70710678]]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">np.linalg.qr(A)</span> - High-performance algorithm computing the decomposition.</li>
                    <li><strong>🤖 ML Connection:</strong> When solving big Machine Learning Least Squares problems `Xw = y`, directly computing `(XᵀX)⁻¹` leads to severe mathematical instability with large floating-point datasets. Advanced solvers heavily utilize QR Decomposition instead because matrix <code>Q</code> has a perfect condition number (stability) of 1!</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="storybook-intro">
        <h2>🔥 Why Orthogonality is Critical in Deep Learning</h2>
        
        <p>In massive deep networks spanning billions of parameters, if weight matrices are perfectly <strong>well-conditioned</strong>:</p>
        <ul>
            <li>✔ Prevents Multicollinearity internally.</li>
            <li>✔ Improves numerical stability in 64-bit and 32-bit floats.</li>
            <li>✔ Massively stabilizes Optimization / gradient descent search curves.</li>
        </ul>
        <p>If they are NOT well-conditioned, the eigenvalues explode or vanish causing <strong>Gradients to Explode violently</strong>. Modern <strong>Orthogonal Initialization</strong> ensures the training state remains pristine and stable.</p>
    </div>

    <h2 class="module-title">Module 10: Singular Value Decomposition (SVD)</h2>

    <div class="storybook-intro">
        <h2>📌 1. What is SVD? (The Backbone of Data Science)</h2>
        
        <p>SVD is the mathematical backbone of PCA, Recommender Systems (like Netflix), Image Compression, and NLP embeddings. For <strong>any matrix A</strong> (even non-square or rank-deficient), SVD decomposes it into three matrices:</p>
        <pre style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; color:#38bdf8;">A = U Σ Vᵀ</pre>
        <ul>
            <li><strong>U</strong> → Left singular vectors (Rotation)</li>
            <li><strong>Σ (Sigma)</strong> → Diagonal matrix of singular values (Scaling/Stretching)</li>
            <li><strong>Vᵀ</strong> → Right singular vectors (Rotation)</li>
        </ul>
        
        <h3 style="margin-top: 1rem;">🔥 Why is it so powerful in ML?</h3>
        <p>Unlike Eigen Decomposition (which only works properly for square, invertible matrices), <strong>SVD works for literally everything</strong>: rectangular matrices, non-invertible matrices, and rank-deficient matrices. It breaks down ANY linear transformation perfectly.</p>
    </div>

    <div class="storybook-intro">
        <h2>📌 2. What are Singular Values?</h2>
        
        <p>Singular values are mathematically the square roots of the eigenvalues of <code>AᵀA</code>. They measure <strong>how much stretching happens</strong> along certain mathematical directions in your dataset.</p>
        
        <h3 style="margin-top: 1rem;">🤖 ML Example: Dataset Matrix</h3>
        <p>Suppose you have a dataset with 3 samples and 2 features:</p>
        <pre style="background: rgba(0,0,0,0.5); padding: 5px; border-radius: 5px; color:#38bdf8;">
[ 1  2 ]
[ 3  4 ]
[ 5  6 ]</pre>
        <p>Running SVD on this exact feature matrix finds the absolute most important feature variance directions without even computing a covariance matrix. Truncating the smallest singular values gives you perfect Low-Rank Approximations (the math behind Image Compression and basic NLP Topic Modeling).</p>
    </div>


    <h2 class="module-title">Module 11: Optimization & ML Connection</h2>

    <div class="storybook-intro">
        <h2>Where Linear Algebra meets Machine Learning Training!</h2>
        <p>This part explains how models fundamentally learn, why gradient descent works, why some problems are perfectly stable, and why some models violently diverge. Since you are working in optimization, MPC, and AI forecasting, this is incredibly critical.</p>
    </div>

    <div class="task-card">
        <h3 class="task-title">Topic 24: Gradient & Gradient Descent</h3>
        <p class="task-desc">The Gradient points in the <strong>direction of steepest increase</strong>. In ML, we move in the opposite direction (Gradient Descent) to minimize loss.</p>
        <button class="btn-toggle" onclick="toggleCode('la-task24', this)">Show Answer</button>
        
        <div id="la-task24" class="answer-container">
            <div class="code-header">
                <div class="window-dots"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
                <button class="copy-btn" onclick="copyToClipboard(this)">Copy Code</button>
            </div>
            <div class="code-container">
<code><span class="keyword">import</span> numpy <span class="keyword">as</span> np

<span class="comment"># Design Matrix (X) and Targets (y)</span>
X = np.array([[1, 1],
              [1, 2],
              [1, 3]])
y = np.array([2, 3, 4])

<span class="comment"># Initial weights</span>
w = np.array([0, 0])

<span class="comment"># Linear Regression Loss Gradient: ∇J = Xᵀ(Xw - y)</span>
gradient = X.T @ (X @ w - y)

print("Loss Gradient Vector:", gradient)</code>
            </div>
            <button class="btn-output" onclick="toggleOutput('out24', this)">Show Output</button>
            <div id="out24" class="output-container">
Loss Gradient Vector: [ -9 -20]
            </div>
            <div class="line-by-line mt-3">
                <strong>📝 Line-by-Line Explanation:</strong>
                <ul>
                    <li><span class="code-snippet">X.T @ (X @ w - y)</span> - This mathematical vector perfectly tells the network exactly how to adjust the weights to reduce error.</li>
                    <li><strong>🤖 Gradient Descent Rule:</strong> <code>w_new = w - η∇J</code> (where <code>η</code> is the learning rate). If the gradient is large, the update is fast. If small, learning slows down!</li>
                </ul>
            </div>
        </div>
    </div>


    <div class="storybook-intro">
        <h2>📐 Hessian Matrix & Quadratic Forms</h2>
        
        <h3 style="margin-top: 1rem;">1. The Hessian Matrix</h3>
        <p>The Hessian is the matrix of <strong>second derivatives</strong>. While the gradient tells you the <em>slope</em>, the Hessian tells you the <strong>Curvature</strong> (whether you are in a bowl minimum, a peak maximum, or a saddle point).</p>
        <p>For Linear Regression quadratic loss, the Hessian is perfectly defined as <code>H = XᵀX</code>! Matrix <code>XᵀX</code> controls the entire curvature of the training landscape.</p>

        <h3 style="margin-top: 1rem;">2. Positive Definite Matrices & Convex Functions</h3>
        <p>When is a function <strong>Convex</strong> (having a perfect, single global minimum)? When its Hessian is strictly <strong>Positive Definite</strong> (all eigenvalues > 0).</p>
        
        <h3 style="margin-top: 1rem;">🤖 ML Connection: Stability</h3>
        <p>In Linear regression, if your features are perfectly independent, <code>XᵀX</code> has eigenvalues > 0. It is positive definite, and a unique solution exists perfectly! But if an eigenvalue = 0, you have multicollinearity! The Hessian is flattened in one direction, creating an infinitely long valley where gradient descent wanders aimlessly (unstable model).</p>
    </div>

    <div class="storybook-intro">
        <h2>🔥 Eigenvalues & Advanced Optimization</h2>
        
        <h3 style="margin-top: 1rem;">Eigenvalues & Conditioning</h3>
        <p>The eigenvalues of the Hessian literally determine the speed of convergence. If the largest eigenvalue is extremely big, the loss landscape is incredibly steep in one direction. If the smallest eigenvalue is small, it is perfectly flat in another.</p>
        
        <p>Condition number: <code>κ = λ_max / λ_min</code>.</p>
        <p><strong>🤖 ML Connection:</strong> If <code>κ</code> is huge, the problem is severely ill-conditioned. Gradient descent struggles violently, causing the notorious exploding or vanishing gradients in Deep Learning!</p>

        <h3 style="margin-top: 1rem;">Newton's Method (MPC / Control Systems)</h3>
        <p>Standard Gradient Descent: <code>w = w - η∇J</code></p>
        <p><strong>Newton's Method:</strong> <code>w = w - H⁻¹∇J</code></p>
        <p>Instead of just blindly stepping down a slope, Newton's method uses the inverse Hessian to calculate the <em>exact curvature</em> and jump directly to the minimum. It is incredibly fast (few steps) but computationally insanely expensive to invert a giant Hessian. <strong>This is the core foundation used in massive advanced optimization solvers and Model Predictive Control (MPC)!</strong></p>
    </div>

    <h2 class="module-title" style="margin-top: 50px;">Conclusion: The Two Lenses of Linear Algebra</h2>
    <p style="text-align: center; color: var(--text-muted); font-size: 1.1rem; max-width: 800px; margin: 0 auto 30px;">To truly master Linear Algebra for ML, you must see it from two distinct mathematical angles: The <strong>Numeric View</strong> and the <strong>Geometric View</strong>. When you possess both, Machine Learning intuitively clicks.</p>

    <div class="storybook-intro" style="border-left: 5px solid #38bdf8;">
        <h2>🔵 1. The Numeric Level (Computation View)</h2>
        <p>This is the world of raw numbers, rigid formulas, matrix operations, Python code, and algorithms. In this level, everything is just blunt mechanical calculations.</p>
        
        <h3 style="margin-top: 1rem;">Example: Matrix Multiplication</h3>
        <p>You have matrix <code>A</code> and matrix <code>B</code>. You painstakingly calculate the dot products of the rows and columns to find the answers: <code>[17, 39]</code>. It is just arithmetic.</p>
        
        <h3 style="margin-top: 1rem;">Example: Linear Regression Formula</h3>
        <p><code>w = (XᵀX)⁻¹Xᵀy</code></p>
        <p>You compute the inverse, multiply it out, and the computer spits out the answer. You are done.</p>

        <h3 style="margin-top: 1rem;">📌 The Takeaway</h3>
        <p>The Numeric Level is <strong>Exactly how a CPU/GPU sees Machine Learning.</strong> Training a neural net is simply: Multiply this matrix, add this bias, compute this floating-point gradient, update these numbers. Pure math.</p>
    </div>

    <div class="storybook-intro" style="border-left: 5px solid #f472b6;">
        <h2>🟣 2. The Geometric Level (Shape & Space View)</h2>
        <p>This is much deeper. Here, matrices are NOT grids of numbers. They are <strong>Transformers of Space</strong>. They dictate the stretching, rotation, projection, and compression of reality.</p>
        
        <h3 style="margin-top: 1rem;">Example: Matrix Multiplication</h3>
        <p>A matrix is a physical transformation. A matrix like <code>[[2,0], [0,1]]</code> does not mean "multiply numbers." It physically means: <em>Stretch the entire dimensional coordinate system across the X-axis by x2, and leave the Y-axis alone.</em> Matrix multiplication is the sheer <strong>deformation of space</strong>.</p>
        
        <h3 style="margin-top: 1rem;">Example: Linear Regression</h3>
        <p>Instead of number crunching <code>(XᵀX)⁻¹Xᵀy</code>, what are we actually doing? We are literally <strong>projecting the target vector `y` straight downwards onto the flat column-space constructed by `X`</strong>. That resulting geometric projection IS the absolute best mathematical approximation. Suddenly, regression makes perfect visual sense.</p>
        
        <h3 style="margin-top: 1rem;">Example: Neural Networks (SVD & Gradients)</h3>
        <ul>
            <li><strong>SVD:</strong> Tells us every single matrix is literally just a <code>Rotate → Stretch → Rotate</code>. So a neural network layer is simply deforming your data space so it's easier to slice!</li>
            <li><strong>Gradient Descent:</strong> You aren't "updating numbers." You are physically walking downhill on a curved, high-dimensional valley surface. The Gradient is the steepest downhill arrow. The Hessian is the bowl's curvature.</li>
        </ul>

        <h3 style="margin-top: 1rem;">📌 The Takeaway</h3>
        <p>The Geometric Level is <strong>Exactly how AI Researchers think.</strong> It focuses purely on structural and conceptual understanding.</p>
    </div>


    <div class="storybook-intro">
        <h2>🔥 Why You Need Both Lenses</h2>
        <div style="overflow-x: auto; margin-top: 1rem;">
            <table class="data-table" style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="padding: 10px; border-bottom: 2px solid rgba(255,255,255,0.2); text-align: left;">Numeric View (Computations)</th>
                        <th style="padding: 10px; border-bottom: 2px solid rgba(255,255,255,0.2); text-align: left;">Geometric View (Shapes)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">Crucial for Python implementation and coding</td>
                        <td style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">Crucial for pure human intuition</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">Excellent for passing mathematics exams</td>
                        <td style="padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">Excellent for deep ML Research and Architecture</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px;">The language of the <strong>Computer</strong></td>
                        <td style="padding: 10px;">The language of the <strong>Researcher</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>


    </div><!-- .container -->

    <script src="script.js?v=2"></script>
</body>
</html>
"""

with open('c:/Users/Mujahid/.gemini/antigravity/scratch/pt/linear-algebra.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
