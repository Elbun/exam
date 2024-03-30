## Exam Socre
# Import Libraries
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import scipy
import math 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 


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
    From the boxplot, we can see the comparation of data distribution in two groups. Assuming 
    the two independent samples are drawn from populations with unequal variances 
    (${{\sigma_A}^2 ≠ {\sigma_B}^2}$), then the test statistic ${t}$ is computed as:
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
    ''' higher writing score mean than Group B. In conclusion, there is difference in all subject score between Group A and Group B.
    ''')



st.subheader("Which major factors contribute to test outcomes?")
st.write('''
    Let's assume all factors (gender, race/ethnicity, parental level of education, lunch, and test preparation course) 
    have an interaction effects on subject score. We will find what factor that has the most contribution on the score. 
    In Machine Learning term, it is called by Feature (Variable) Importance.
         
    Feature (variable) importance indicates how much each feature (variabel) contributes to the model prediction. 
    Basically, it determines the degree of usefulness of a specific variable for a current model and prediction.
    In this case, we want to predict the subject score based on the all factors.
         
    Based on the dataset, we will use Linear Regression method to predict the each subject score with all factors. 
    Using this method, the feature (variable) importance would be the coefficient of each factor.
    Let's start with predict the math score. All factors will be grouped as ${X}$ (independent variables) and the math score 
    will be grouped as ${y}$ (dependent variable). First, we need to convert all factors value to numerical data so they can 
    be computed in the model. 
    ''')

col1, col2 = st.columns(2)
with col1:
    df = pd.DataFrame(dataset['gender'].unique())
    df = df.rename(columns={0: "gender"})
    custom_dict = {'male':0, 'female':1}
    df = df.sort_values(by=['gender'], key=lambda x: x.map(custom_dict))
    df['gender_convert'] = np.arange(df.shape[0])+1
    df = df.set_index(df['gender_convert'])
    st.table(df)
    dataset_join = pd.merge(dataset, df, how='left', on='gender')

    df = pd.DataFrame(dataset['race/ethnicity'].unique())
    df = df.rename(columns={0: "race/ethnicity"})
    custom_dict = {'group A':0, 'group B':1, 'group C':2, 'group D':3, 'group E':4}
    df = df.sort_values(by=['race/ethnicity'], key=lambda x: x.map(custom_dict))
    df['race/ethnicity_convert'] = np.arange(df.shape[0])+1
    df = df.set_index(df['race/ethnicity_convert'])
    st.table(df)
    dataset_join = pd.merge(dataset_join, df, how='left', on='race/ethnicity')

    df = pd.DataFrame(dataset['lunch'].unique())
    df = df.rename(columns={0: "lunch"})
    custom_dict = {'free/reduced':0, 'standard':1}
    df = df.sort_values(by=['lunch'], key=lambda x: x.map(custom_dict))
    df['lunch_convert'] = np.arange(df.shape[0])+1
    df = df.set_index(df['lunch_convert'])
    st.table(df)
    dataset_join = pd.merge(dataset_join, df, how='left', on='lunch')

with col2:
    df = pd.DataFrame(dataset['parental level of education'].unique())
    df = df.rename(columns={0: "parental level of education"})
    custom_dict = {'some high school':0, 'high school':1, 'some college':2, 'associate\'s degree':3, 'bachelor\'s degree':4, 'master\'s degree':5}
    df = df.sort_values(by=['parental level of education'], key=lambda x: x.map(custom_dict))
    df['parental level of education_convert'] = np.arange(df.shape[0])+1
    df = df.set_index(df['parental level of education_convert'])
    st.table(df)
    dataset_join = pd.merge(dataset_join, df, how='left', on='parental level of education')

    df = pd.DataFrame(dataset['test preparation course'].unique())
    df = df.rename(columns={0: "test preparation course"})
    custom_dict = {'none':0, 'completed':1}
    df = df.sort_values(by=['test preparation course'], key=lambda x: x.map(custom_dict))
    df['test preparation course_convert'] = np.arange(df.shape[0])+1
    df = df.set_index(df['test preparation course_convert'])
    st.table(df)
    dataset_join = pd.merge(dataset_join, df, how='left', on='test preparation course')

st.write('''Table to be (sample data) :''')
st.table(dataset_join.head(10))

st.write('''
    We will split the dataset into train dataset and test dataset. The train data set is used to train the prediction model.
    The test dataset will be used to evaluate the prediction output. The dataset will be splitted 80:20 for train and test
    dataset. The size of each dataset is shown as below.
    ''')
X = dataset_join[['gender_convert','race/ethnicity_convert','parental level of education_convert','lunch_convert','test preparation course_convert']]
Y = dataset_join['math score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
col1, col2 = st.columns(2)
with col1:
    st.write('${X_{train}}$ = ', X_train.shape)
    st.write('${X_{test}}$ = ', X_test.shape)
with col2:
    st.write('${y_{train}}$ = ', y_train.shape)
    st.write('${y_{test}}$ = ', y_test.shape)

st.write('''
    Now we use ${X_{train}}$ and ${y_{train}}$ to train the linear regression model. After the model is built, one of parameters
    we can find is intercept. Intercept is the value of ${y}$ where all ${X}$ values are zero.
    ''')
model = LinearRegression()
model.fit(X_train,y_train)
st.write('Intercept = ', model.intercept_)

st.write('''
    The other parameter we can find is coefficient. The sign of each coefficient indicates the direction of the relationship between 
    a predictor variable (${X}$) and the response variable (${y}$). A positive sign indicates that as the predictor variable increases, 
    the target variable also increases. A negative sign indicates that as the predictor variable increases, the target variable decreases.
    From the coefficient of each variable, the most contribution to the math score is given by lunch variable with standard value.
    Standar value in lunch variable is predicted can increase the math score by 11.7268 points.
    ''')
coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
st.table(coeff_parameter)

st.write('''
    After knowing the intercept and coefficient of each ${X}$, we can start to predict the math score from ${X_{test}}$ using equaition model.
    Each predicted math score (${y_{pred}}$) is compared with the actual math score (${y_{test}}$) and then plotted in scatter chart.
    ''')
y_pred = model.predict(X_test)
mse_label = mean_squared_error(y_test, y_pred)

## Chart 
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
scatter1 = alt.Chart(df).mark_point(size=20).encode(
    x=alt.X("y_test:Q", title="y_test").scale(zero=True),
    y=alt.Y("y_pred:Q", title="y_pred").scale(zero=True)
    )
final_plot = scatter1 + scatter1.transform_regression("y_test","y_pred").mark_line(color="red")
fig = (final_plot).configure_axis(
        labelFontSize=10
    ).properties(
        title=('Actual and Predicted Math Score Comparison'),
        width=400,
        height=400
    )
st.altair_chart(fig)

st.write('''
    Let's evaluate the other subject score with the same prediction method. Then compare the each other value.
    ''')
col1, col2, col3 = st.columns(3)
with col1:
    st.write('Math Score Prediction')
    Y_math = dataset_join['math score']
    X_train, X_test, y_train, y_test = train_test_split(X, Y_math, test_size=0.2, random_state=101)
    model.fit(X_train,y_train)
    intercept_math = model.intercept_
    st.write('Intercept = ', intercept_math)
    coeff_math = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient (Math)'])
    mse_math = mean_squared_error(y_test, y_pred)
    r2_math = model.score(X_test, y_test)
    
    ## Chart 
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    scatter1 = alt.Chart(df).mark_point(size=20).encode(
        x=alt.X("y_test:Q", title="y_test").scale(zero=True),
        y=alt.Y("y_pred:Q", title="y_pred").scale(zero=True)
        )
    final_plot = scatter1 + scatter1.transform_regression("y_test","y_pred").mark_line(color="red")
    fig_math = (final_plot).configure_axis(
            labelFontSize=10
        ).properties(
            title=('Actual vs Predicted Math Score'),
            width=300,
            height=300
        )
    
with col2:
    st.write('Reading Score Prediction')
    Y_reading = dataset_join['reading score']
    X_train, X_test, y_train, y_test = train_test_split(X, Y_reading, test_size=0.2, random_state=101)
    model.fit(X_train,y_train)
    intercept_reading = model.intercept_
    st.write('Intercept = ', intercept_reading)
    coeff_reading = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient (Reading)'])
    mse_reading = mean_squared_error(y_test, y_pred)
    r2_reading = model.score(X_test, y_test)
    
    ## Chart 
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    scatter1 = alt.Chart(df).mark_point(size=20).encode(
        x=alt.X("y_test:Q", title="y_test").scale(zero=True),
        y=alt.Y("y_pred:Q", title="y_pred").scale(zero=True)
        )
    final_plot = scatter1 + scatter1.transform_regression("y_test","y_pred").mark_line(color="red")
    fig_reading = (final_plot).configure_axis(
            labelFontSize=10
        ).properties(
            title=('Actual vs Predicted Reading Score'),
            width=300,
            height=300
        )
    
with col3:
    st.write('Writing Score Prediction')
    Y_writing = dataset_join['writing score']
    X_train, X_test, y_train, y_test = train_test_split(X, Y_writing, test_size=0.2, random_state=101)
    model.fit(X_train,y_train)
    intercept_writing = model.intercept_
    st.write('Intercept = ', intercept_writing)
    coeff_writing = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient (Writing)'])
    mse_writing = mean_squared_error(y_test, y_pred)
    r2_writing = model.score(X_test, y_test)
    
    ## Chart 
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    scatter1 = alt.Chart(df).mark_point(size=20).encode(
        x=alt.X("y_test:Q", title="y_test").scale(zero=True),
        y=alt.Y("y_pred:Q", title="y_pred").scale(zero=True)
        )
    final_plot = scatter1 + scatter1.transform_regression("y_test","y_pred").mark_line(color="red")
    fig_writing = (final_plot).configure_axis(
            labelFontSize=10
        ).properties(
            title=('Actual vs Predicted Writing Score'),
            width=300,
            height=300
        )

coeff_all = coeff_math.join(coeff_reading)
coeff_all = coeff_all.join(coeff_writing)
st.table(coeff_all)

col1, col2 = st.columns(2)
with col1:
    st.altair_chart(fig_math)
with col2:
    st.altair_chart(fig_reading)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(fig_writing)
with col2:
    st.write('Mean Squared Error and ${R^2}$')
    st.write('MSE (Math) = ', mse_math)
    st.write('MSE (Reading) = ', mse_reading)
    st.write('MSE (Writing) = ', mse_writing)
    st.write('${R^2}$ (Math) = ', r2_math)
    st.write('${R^2}$ (Reading) = ', r2_reading)
    st.write('${R^2}$ (Writing) = ', r2_writing)

st.write('''
    These three prediction models doesn't seems really good at predict the subject score with all factors as
    the model input. If we look carefully at the graphs, we found that the ${y_{pred}}$ values are never fall below 40.
    While in the actual condition ${y_{test}}$, there are many value of score fall below 40.
         
    Mean Squared Error (MSE) is a commonly used metric for evaluating the performance of regression models. The MSE of all
    three models is higher then 100 while the subject score only available between 0 to 100. This condition may indicate 
    substantial prediction errors.
         
    Coefficient of Determination (${R^2}$) represents the proportion of the variance in the dependent variable that is 
    explained by the independent variables in a regression model. It ranges from 0 to 1 and is often expressed as a percentage.
    None of three models has ${R^2}$ greater than 50%. So the models can be good enough to predict the subject score.
    There must be other factors besides the factors before that contributes in the subject score prediction.
         
    This low performance of regression model may be caused by some irrelevant or less-important variable as the input.
    Look at the coefficient comparison of the models. The top 3 factors are lunch, test preparation course, and gender.
    Since gender is considered as an uncontrollable factor, it should excluded as factor that determines someone's subject 
    score. So the most importance factors that determine the subject score are lunch and test preparation course.
    ''')




st.subheader("What would be the best way to improve student scores on each test?")
st.write('''
    From the first questions before, we know that the subjects' score from student in Group A (completed the test preparation course)
    are statistically different with the score in Group B (not completed the test preparation course). Group A's score mean
    are higher than the Group B's. This indicates that taking test preparation course before exam has positive effect on the 
    exam result.
    
    Answer for the second question shows that there are two major factors that is predicted contribute in subject score. They are
    lunch and test preparation course. In the prediction model building process, we found that the models doesn't seems
    really good at predicting the subject score. The highest ${R^2}$ calculated is only 41.4% which is still below 50%. There must
    be other factors contribute in determining the exam result (subject score).
    
    In conclusion, from the analysis, get a standar lunch and complete the test preparation course are good ways to improve 
    each subject score. But there are also other additional way to improve them, such as take an additional study hour or create a
    different teaching method (for example a group learning). The learning environment and exam environment also can be considered 
    as one factor that can affect the exam result.
    ''')