# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:50:09 2024

@author: jcp
"""

import prep
import visual
import prep_pt as pp

prep.make_data('경기')

visual.make_charts('경기')
visual.make_charts('서울')
# %%
pp.make_data('세종')
visual.make_charts('세종')

# %%
prep.make_data('인천')
visual.make_charts('인천')
# %%
prep.make_data('대구')
visual.make_charts('대구')
# %%
cities = ['서울','경기','부산','대구','인천','세종','강원','울산','광주','대전','제주','경북','경남','전북','전남','충북','충남']
len(cities)
for city in cities:
    pp.make_data(city)
    visual.make_charts(city)