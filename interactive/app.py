from flask import Flask, request, render_template, redirect
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import math

df = pd.read_csv('../foreveralone.csv')
l = ['gender','income','bodyweight','social_fear','attempt_suicide','employment','edu_level']
restricted_df = df[l]

N,C = restricted_df.shape
for i in range(N):
	for j in range(C):
		if restricted_df.iloc[i,j] in ['Male','Yes','Transgender male']: 
			restricted_df.set_value(i, restricted_df.columns[j], 1)
		elif restricted_df.iloc[i,j] in ['Female','No','Transgender female']:
			restricted_df.set_value(i, restricted_df.columns[j], 0)

gender = restricted_df['gender']
income = pd.get_dummies(restricted_df['income'])
income.drop(income.columns[len(income.columns)-1], axis=1, inplace=True)
bodyweight = pd.get_dummies(restricted_df['bodyweight'])
bodyweight.drop(bodyweight.columns[len(bodyweight.columns)-1], axis=1, inplace=True)
social_fear = restricted_df['social_fear']
attempt_suicide = restricted_df['attempt_suicide']
employment = pd.get_dummies(restricted_df['employment'])
employment.drop(employment.columns[len(employment.columns)-1], axis=1, inplace=True)
edu_level = pd.get_dummies(restricted_df['edu_level'])
edu_level.drop(edu_level.columns[len(edu_level.columns)-1], axis=1, inplace=True)

inputdf = pd.concat([gender,income,bodyweight,social_fear,attempt_suicide,employment,edu_level],axis=1)
data_mat = inputdf.values.astype('float')

depressed_vec = pd.get_dummies(df['depressed'])['Yes'].values.astype('float')

model = sm.Logit(endog=depressed_vec,exog=sm.add_constant(data_mat))
results = model.fit(method='powell')

def getVal(dat,predict):
	src = predict[dat]
	ret = [0]*10
	for i in src:
		idx = math.floor(i*10)
		if idx == 10:
			idx = 9
		ret[idx] = ret[idx]+1
	ret = [x / sum(ret) for x in ret]
	return ret

datainput = sm.add_constant(data_mat)
predict = results.predict(exog=datainput)
predict = np.array([1-x for x in predict])
dataTable = []
data_mat = data_mat.tolist()
def addData(start, end):
	for j in range(start,end+1):
		if j != end:
			add = []
			select = [x for ind, x in enumerate(data_mat) if x[j]==1]
			for i in select:
				add.append(data_mat.index(i))
			dataTable.append(getVal(add,predict))
		else:
			add = []
			select = [x for ind, x in enumerate(data_mat) if all(x[k] == 0 for k in range(start,end))]
			for i in select:
				add.append(data_mat.index(i))
			dataTable.append(getVal(add,predict))


# gender
addData(0,1)

# Income
addData(1,13)

# bodyweight
addData(13,16)

#socialfear 16
addData(16,17)

#suicide 17
addData(17,18)

#employment 18-25
addData(18,26)

#edu 26-33
addData(26,34)


app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def index():
	if request.method == 'POST':
		# Get input
		result = {}
		result['gender'] = int(request.form['gender'])
		result['socialFear'] = int(request.form['socialFear'])
		result['suicide'] = int(request.form['suicide'])
		result['employment'] = int(request.form['employment'])
		result['edu'] = int(request.form['edu'])
		result['income'] = int(request.form['income'])
		result['bodyweight'] = int(request.form['bodyweight'])

		# record user input
		user = []
		for k,v in result.items():
			user.append(v)
		udat = dict(data = user)

		#generate predict input
		inputval = [0]*34
		inputval[0] = result['gender']
		if result['income'] != -1:
			inputval[result['income']] = 1
		if result['bodyweight'] != -1:
			inputval[result['bodyweight']] = 1
		inputval[16] = result['socialFear']
		inputval[17] = result['suicide']
		if result['employment'] != -1:
			inputval[result['employment']] = 1
		if result['edu'] != -1:
			inputval[result['edu']] = 1
		inputval = [0.6829]+inputval

		score = results.predict(exog=inputval)
		score = (1-score)*100

		global dataTable
		inputDat = dict(dataT = dataTable)


		print(score)
		return render_template("result.html", score = score, **udat, **inputDat)
	return render_template("temp.html") 

