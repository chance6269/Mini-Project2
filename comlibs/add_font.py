# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:54:59 2024

@author: pjc62
"""

def reg():
    import matplotlib.font_manager as fm
    import os

    user_name = os.getlogin()

    fontpath = [f'C:/Users/{user_name}/AppData/Local/Microsoft/Windows/Fonts']
    font_files = fm.findSystemFonts(fontpaths=fontpath)
    for fpath in font_files:
        
        fm.fontManager.addfont(fpath)
        
# %%

def fontpath():
    
    import os

    user_name = os.getlogin()

    return f'C:/Users/{user_name}/AppData/Local/Microsoft/Windows/Fonts'