# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:50:09 2024

@author: jcp
"""

import prep
import visual

prep.make_data('경기')

visual.make_charts('경기')
visual.make_charts('서울')
# %%
prep.make_data('세종')
visual.make_charts('세종')

# %%
prep.make_data('인천')
visual.make_charts('인천')