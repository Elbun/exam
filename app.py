## Exam Socre
# Import Libraries
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import scipy
import math


# Get the data
dataset = pd.read_csv("exams.csv")


# Set page config
st.set_page_config(
    page_title="Student Exam Score Evaluation",
    page_icon="🚹"
)
st.title("Student Exam Score Evaluation")

st.header("Overview")
st.write('''
    This is a fictional open dataset and will only be used for data science training purposes here.
    This dataset includes scores from three exams and a variety of personal, social, and economic 
    factors that have interaction effects upon them. This dataset can be obtained from the source below.
    ''')
st.link_button("Data Source Link", "http://roycekimmons.com/tools/generated_data/exams")

st.header("Sample Dataset")
st.table(dataset.head(10))

st.header("Data Analysis")
st.write('''
    This section will show the data analysis process to answer several question.
    - How effective is the test preparation course?
    - Which major factors contribute to test outcomes?
    - What would be the best way to improve student scores on each test?
         
    Let's try to answer the first question.
    ''')

st.subheader("How effective is the test preparation course?")
st.write('''
    We will analyze each subject score with only 'test preparation course' as the grouping variable. 
    We call the group who completed the test preparation course as Group A and the group who didn't
    as Group B. This analysis will use independent samples T-Test as the method. We will set the 
    null hypothesis (${H_0}$) as there is no significant score mean difference between
    Group A (${\mu_A}$) and Group B (${\mu_B}$). Let's start with the math subject score.
    ''')
st.latex(r'''H_0 : \mu_A - \mu_B = 0''')
st.latex(r'''H_1 : \mu_A - \mu_B \not = 0''')


st.write('''
    Focus on math score in each group then we will get this summary.
    ''')

## Boxplot
box1 = alt.Chart(dataset).mark_boxplot(size=50, extent=0.5).encode(
    x=alt.X("math score:Q", title="Math Score"),
    y=alt.Y("test preparation course:N", title=None, axis=None),
    color="test preparation course:N"
)
fig = (box1).configure_axis(
            labelFontSize=10
        ).properties(
            title=('Math Score Comparison'),
            width=700,
            height=250
        )
st.altair_chart(fig)

na = dataset[dataset["test preparation course"]=="completed"]["math score"].count()
xa = dataset[dataset["test preparation course"]=="completed"]["math score"].mean()
sa = dataset[dataset["test preparation course"]=="completed"]["math score"].std()
nb = dataset[dataset["test preparation course"]=="none"]["math score"].count()
xb = dataset[dataset["test preparation course"]=="none"]["math score"].mean()
sb = dataset[dataset["test preparation course"]=="none"]["math score"].std()

col1, col2, col3 = st.columns([1,1,3])
with col1:
    st.write('Group A')
    st.write('${n_A}$ = ', na)
    st.write('${\overline{x}_A}$ = ', round(xa,2))
    st.write('${s_A}$ = ', round(sa,2))
with col2:
    st.write('Group B')
    st.write('${n_B}$ = ', nb)
    st.write('${\overline{x}_B}$ = ', round(xb,2))
    st.write('${s_B}$ = ', round(sb,2))
with col3:
    st.write('Note')
    st.write('${n_i}$ = data sample size of Group ${i}$')
    st.write('${\overline{x}_i}$ = math score mean of Group ${i}$')
    st.write('${s_i}$ = sample standard deviation of Group ${i}$')

st.write('''
    From the boxplot, assuming the two independent samples are drawn from populations with unequal variances 
    (${{\sigma_A}^2 ≠ {\sigma_B}^2}$), the test statistic ${t}$ is computed as:
    ''')
st.latex(r'''
    t = \frac{\overline{x}_A - \overline{x}_B}{\sqrt{\frac{{s_A}^2}{n_A} + \frac{{s_B}^2}{n_B}}}
    ''')
t = (xa-xb)/(math.sqrt(sa*sa/na + sb*sb/nb))

st.write('''
    The calculated ${t}$ value is then compared to the critical ${t}$ value from the ${t}$ distribution table with degrees of freedom
    (${df}$) and chosen confidence level. If the ${t_{calculated}}$ value > ${t_{critical}}$ value, then we reject the null 
    hypothesis (${H_0}$).
    ''')
st.latex(r'''
    df = \frac{\left(\frac{{s_A}^2}{n_A} + \frac{{s_B}^2}{n_B}\right)^2}
         {\frac{1}{n_A-1}\left(\frac{{s_A}^2}{n_A}\right)^2 + \frac{1}{n_B-1}\left(\frac{{s_B}^2}{n_B}\right)^2}
    ''')
df = ((sa*sa/na + sb*sb/nb)**2) / ((1/(na-1))*((sa*sa/na)**2) + (1/(nb-1))*((sb*sb/nb)**2))

st.write('''
    Using these formulas, we can calculate the value of calculated ${t}$ value and ${df}$.
    ''')
col1, col2 = st.columns(2)
with col1:
    st.write('${t_{calculated}}$ = ', t)
with col2:
    st.write('${df}$ = ', df)


st.write('''
    Choose the confindence level of 5% (two tails) and with the ${df}$ calculated, critical ${t}$ value will be 
    ''')
t_crit = scipy.stats.t.ppf(1-.05/2, df)
st.write('${t_{critical}}$ = ', t_crit)

xab = xa/xb-1

st.write('''
    Because the ${t_{calculated}}$ > ${t_{critical}}$, so the null hypothesis (${H_0}$) is rejected. In other words,
    statistically there is a significant math score mean differences between Group A and Group B. Group A has ''',
    format(xab, ".2%"),
    ''' higher
    math score mean than Group B.

    Now let's use the same method to evaluate the reading score and writing score of two groups.
    ''')


## Boxplot
box1 = alt.Chart(dataset).mark_boxplot(size=50, extent=0.5).encode(
    x=alt.X("reading score:Q", title="Reading Score"),
    y=alt.Y("test preparation course:N", title=None, axis=None),
    color="test preparation course:N"
)
fig = (box1).configure_axis(
            labelFontSize=10
        ).properties(
            title=('Reading Score Comparison'),
            width=700,
            height=250
        )
st.altair_chart(fig)

## Boxplot
box1 = alt.Chart(dataset).mark_boxplot(size=50, extent=0.5).encode(
    x=alt.X("writing score:Q", title="Writing Score"),
    y=alt.Y("test preparation course:N", title=None, axis=None),
    color="test preparation course:N"
)
fig = (box1).configure_axis(
            labelFontSize=10
        ).properties(
            title=('Writing Score Comparison'),
            width=700,
            height=250
        )
st.altair_chart(fig)


col1, col2 = st.columns(2)
with col1:
    st.write('Reading Score Parameter')
    na = dataset[dataset["test preparation course"]=="completed"]["reading score"].count()
    xa = dataset[dataset["test preparation course"]=="completed"]["reading score"].mean()
    sa = dataset[dataset["test preparation course"]=="completed"]["reading score"].std()
    nb = dataset[dataset["test preparation course"]=="none"]["reading score"].count()
    xb = dataset[dataset["test preparation course"]=="none"]["reading score"].mean()
    sb = dataset[dataset["test preparation course"]=="none"]["reading score"].std()
    t = (xa-xb)/(math.sqrt(sa*sa/na + sb*sb/nb))
    df = ((sa*sa/na + sb*sb/nb)**2) / ((1/(na-1))*((sa*sa/na)**2) + (1/(nb-1))*((sb*sb/nb)**2))
    t_crit = scipy.stats.t.ppf(1-.05/2, df)
    xab_reading = xa/xb-1
    col3, col4 = st.columns(2)
    with col3:
        st.write('Group A (completed)')
        st.write('${n_A}$ = ', na)
        st.write('${\overline{x}_A}$ = ', round(xa,2))
        st.write('${s_A}$ = ', round(sa,2))
    with col4:
        st.write('Group B (none)')
        st.write('${n_B}$ = ', nb)
        st.write('${\overline{x}_B}$ = ', round(xb,2))
        st.write('${s_B}$ = ', round(sb,2))
    st.write('${t_{calculated}}$ = ', t)
    st.write('${df}$ = ', df)
    st.write('${t_{critical}}$ = ', t_crit)
with col2:
    st.write('Writing Score Parameter')
    na = dataset[dataset["test preparation course"]=="completed"]["writing score"].count()
    xa = dataset[dataset["test preparation course"]=="completed"]["writing score"].mean()
    sa = dataset[dataset["test preparation course"]=="completed"]["writing score"].std()
    nb = dataset[dataset["test preparation course"]=="none"]["writing score"].count()
    xb = dataset[dataset["test preparation course"]=="none"]["writing score"].mean()
    sb = dataset[dataset["test preparation course"]=="none"]["writing score"].std()
    t = (xa-xb)/(math.sqrt(sa*sa/na + sb*sb/nb))
    df = ((sa*sa/na + sb*sb/nb)**2) / ((1/(na-1))*((sa*sa/na)**2) + (1/(nb-1))*((sb*sb/nb)**2))
    t_crit = scipy.stats.t.ppf(1-.05/2, df)
    xab_writing = xa/xb-1
    col3, col4 = st.columns(2)
    with col3:
        st.write('Group A (completed)')
        st.write('${n_A}$ = ', na)
        st.write('${\overline{x}_A}$ = ', round(xa,2))
        st.write('${s_A}$ = ', round(sa,2))
    with col4:
        st.write('Group B (none)')
        st.write('${n_B}$ = ', nb)
        st.write('${\overline{x}_B}$ = ', round(xb,2))
        st.write('${s_B}$ = ', round(sb,2))
    st.write('${t_{calculated}}$ = ', t)
    st.write('${df}$ = ', df)
    st.write('${t_{critical}}$ = ', t_crit)
    
st.write('''
    Because the ${t_{calculated}}$ > ${t_{critical}}$ in both reading and writing score, so the null hypothesis (${H_0}$) 
    is rejected. In other words, statistically there is a significant reading and writing score mean differences between 
    Group A and Group B. Group A has ''',
    format(xab_reading, ".2%"),
    ''' higher reading score mean and ''',
    format(xab_writing, ".2%"),
    ''' 
    higher writing score mean than Group B. In conclusion, there is difference in all subject score between Group A and Group B.
    ''')