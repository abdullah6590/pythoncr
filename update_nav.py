import glob
import re

nav_template = """    <nav class="top-nav">
        <div class="dropdown">
            <button class="dropbtn[__PY_ACTIVE__]">�� Python <span class="caret">▼</span></button>
            <div class="dropdown-content">
                <a href="basics.html"[__BA_ACTIVE__]>📘 Basics</a>
                <a href="index.html"[__FN_ACTIVE__]>🐍 Functions</a>
                <a href="oop.html"[__OO_ACTIVE__]>🏗️ OOP</a>
            </div>
        </div>
        <div class="dropdown">
            <button class="dropbtn[__LI_ACTIVE__]">📚 Libraries <span class="caret">▼</span></button>
            <div class="dropdown-content">
                <a href="numpy.html"[__NU_ACTIVE__]>🧮 NumPy</a>
                <a href="pandas.html"[__PA_ACTIVE__]>🐼 Pandas</a>
                <a href="matplotlib.html"[__MA_ACTIVE__]>📊 Matplotlib</a>
            </div>
        </div>
        <div class="dropdown">
            <button class="dropbtn[__ML_ACTIVE__]">🤖 Machine Learning <span class="caret">▼</span></button>
            <div class="dropdown-content">
                <a href="ml-intro.html"[__IN_ACTIVE__]>🚀 Introduction</a>
                <a href="ml-supervised.html"[__SU_ACTIVE__]>📊 Supervised Learning</a>
                <a href="ml-classification-regression.html"[__CR_ACTIVE__]>🎯 Classification / Regression</a>
                <a href="ml-ensemble.html"[__EN_ACTIVE__]>🤝 Ensemble Learning</a>
                <a href="ml-semi-supervised.html"[__SE_ACTIVE__]>🔀 Semi-Supervised Learning</a>
                <a href="ml-unsupervised.html"[__UN_ACTIVE__]>🔍 Unsupervised Learning</a>
                <a href="ml-dimensionality-reduction.html"[__DR_ACTIVE__]>📉 Dimensionality Reduction</a>
                <a href="ml-reinforcement.html"[__RE_ACTIVE__]>🎮 Reinforcement Learning</a>
            </div>
        </div>
        <div class="dropdown">
            <button class="dropbtn[__DL_ACTIVE__]">🧠 Deep Learning <span class="caret">▼</span></button>
            <div class="dropdown-content">
                <a href="dl-intro.html"[__DLIN_ACTIVE__]>🚀 Introduction</a>
            </div>
        </div>
        <div class="dropdown">
            <button class="dropbtn[__MA_BTN_ACTIVE__]">➗ Mathematics <span class="caret">▼</span></button>
            <div class="dropdown-content">
                <a href="linear-algebra.html"[__LA_ACTIVE__]>📐 Linear Algebra</a>
                <a href="probability.html"[__PR_ACTIVE__]>🎲 Probability</a>
                <a href="statistics.html"[__ST_ACTIVE__]>📊 Statistics</a>
                <a href="calculus.html"[__CA_ACTIVE__]>📈 Calculus</a>
            </div>
        </div>
    </nav>"""

active_map = {
    'basics.html': ('__PY_ACTIVE__', '__BA_ACTIVE__'),
    'index.html': ('__PY_ACTIVE__', '__FN_ACTIVE__'),
    'oop.html': ('__PY_ACTIVE__', '__OO_ACTIVE__'),
    
    'numpy.html': ('__LI_ACTIVE__', '__NU_ACTIVE__'),
    'pandas.html': ('__LI_ACTIVE__', '__PA_ACTIVE__'),
    'matplotlib.html': ('__LI_ACTIVE__', '__MA_ACTIVE__'),
    
    'ml-intro.html': ('__ML_ACTIVE__', '__IN_ACTIVE__'),
    'ml-supervised.html': ('__ML_ACTIVE__', '__SU_ACTIVE__'),
    'ml-classification-regression.html': ('__ML_ACTIVE__', '__CR_ACTIVE__'),
    'ml-ensemble.html': ('__ML_ACTIVE__', '__EN_ACTIVE__'),
    'ml-semi-supervised.html': ('__ML_ACTIVE__', '__SE_ACTIVE__'),
    'ml-unsupervised.html': ('__ML_ACTIVE__', '__UN_ACTIVE__'),
    'ml-dimensionality-reduction.html': ('__ML_ACTIVE__', '__DR_ACTIVE__'),
    'ml-reinforcement.html': ('__ML_ACTIVE__', '__RE_ACTIVE__'),
    
    'dl-intro.html': ('__DL_ACTIVE__', '__DLIN_ACTIVE__'),
    
    'linear-algebra.html': ('__MA_BTN_ACTIVE__', '__LA_ACTIVE__'),
    'probability.html': ('__MA_BTN_ACTIVE__', '__PR_ACTIVE__'),
    'statistics.html': ('__MA_BTN_ACTIVE__', '__ST_ACTIVE__'),
    'calculus.html': ('__MA_BTN_ACTIVE__', '__CA_ACTIVE__')
}

files = glob.glob('*.html')
nav_pattern = re.compile(r'<nav class="top-nav">.*?</nav>', re.DOTALL)

for f in files:
    if f not in active_map:
        continue
    
    with open(f, 'r') as file:
        content = file.read()
    
    # Generate customized nav for this file
    custom_nav = nav_template
    # 1. Reset all to empty
    for token in ['__PY_ACTIVE__', '__BA_ACTIVE__', '__FN_ACTIVE__', '__OO_ACTIVE__', 
                  '__LI_ACTIVE__', '__NU_ACTIVE__', '__PA_ACTIVE__', '__MA_ACTIVE__', 
                  '__ML_ACTIVE__', '__IN_ACTIVE__', '__SU_ACTIVE__', '__CR_ACTIVE__', 
                  '__EN_ACTIVE__', '__SE_ACTIVE__', '__UN_ACTIVE__', '__DR_ACTIVE__', '__RE_ACTIVE__',
                  '__DL_ACTIVE__', '__DLIN_ACTIVE__',
                  '__MA_BTN_ACTIVE__', '__LA_ACTIVE__', '__PR_ACTIVE__', '__ST_ACTIVE__', '__CA_ACTIVE__']:
        custom_nav = custom_nav.replace(token, '')
        
    # 2. Add active classes
    btn_active, sub_active = active_map[f]
    # Re-inject the token that needs ' active' instead of empty, but since we already deleted them...
    # Better approach: string formatting or just raw replace. 
    # Let's recreate custom_nav again correctly
    pass

for f in files:
    if f not in active_map:
        continue
    
    with open(f, 'r') as file:
        content = file.read()
    
    custom_nav = nav_template
    btn_active, sub_active = active_map[f]
    
    # Replace active tokens
    custom_nav = custom_nav.replace(btn_active, ' active')
    custom_nav = custom_nav.replace(sub_active, ' class="active"')
    
    # Clear remaining tokens
    for t in active_map.values():
        custom_nav = custom_nav.replace(t[0], '')
        custom_nav = custom_nav.replace(t[1], '')
    
    # Do replacement!
    new_content = nav_pattern.sub(custom_nav, content)
    
    with open(f, 'w') as file:
        file.write(new_content)

print(f"Updated navbars in {len(active_map)} files!")
