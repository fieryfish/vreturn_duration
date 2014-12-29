This class is to calculate the Return duration which consist of audit and
production in each inventory.
This model use the RandomForestRegressor to train the model. For the
categorical feature, we use OneHotEncoder techique to do the preprocessing.
For the audit part, we only the the FirDeptID field to train the model which
is not so precise because of the low R square value. But the audit duration is not quite long, so it doesn't affect too much
For the production part, we use 'ReturnCompanyID','StockNo' and 'FactNum' to
train the model which means '配送中心','库房' and '实际退货数量'.

