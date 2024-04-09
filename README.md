## Modelos de Clasificación con Balanceo de Datos

Este repositorio contiene código para entrenar y evaluar modelos de clasificación utilizando técnicas de balanceo de datos. Se incluyen ejemplos de modelado sin balanceo, así como modelos con datos balanceados mediante las técnicas de oversampling y undersampling.

## Contenido
- Modelado de Datos Sin Balanceo
- Modelos con Datos Balanceados por Oversampling
- Modelos con Datos Balanceados por Undersampling
- Modelado de Datos Sin Balanceo
  
En esta sección, se encuentra el código para entrenar y evaluar modelos de clasificación sin aplicar técnicas de balanceo de datos.

**Modelos con Datos Balanceados por Oversampling**

Aquí se presentan ejemplos de modelos entrenados con datos balanceados mediante la técnica de oversampling. Se utiliza la clase RandomOverSampler para aumentar la cantidad de muestras de la clase minoritaria.

**Modelos con Datos Balanceados por Undersampling**

En esta parte, se muestran modelos entrenados con datos balanceados mediante la técnica de undersampling. Se emplea la clase RandomUnderSampler para reducir la cantidad de muestras de la clase mayoritaria.
Ejemplo de Código
A continuación, se presenta un ejemplo de código utilizado para entrenar y evaluar modelos con datos balanceados por undersampling:
```
X, y = x_train, y_train
random_ud = RandomUnderSampler()
X_ran, y_ran = random_ud.fit_resample(X, y)

print(y_ran.value_counts() * 100 / len(y_ran))
print(y_ran.value_counts())

lm = LogisticRegression(max_iter=150)
tree = DecisionTreeClassifier(random_state=42)

lm.fit(X_ran, y_ran)
tree.fit(X_ran, y_ran)

lm_predict = lm.predict(x_test)
tree_predict = tree.predict(x_test)

lm_predict_prob = lm.predict_proba(x_test)[:, 1]
tree_predict_prob = tree.predict_proba(x_test)[:, 1]

print('**Logística**')
print(classification_report(y_test, lm_predict))

print('**Árboles**')
print(classification_report(y_test, tree_predict))

print('AUC Logística', roc_auc_score(y_test, lm_predict))
print('AUC Árboles', roc_auc_score(y_test, tree_predict))
```

Este código entrena modelos de regresión logística y árboles de decisión utilizando datos balanceados por undersampling. Luego, evalúa el desempeño de los modelos utilizando métricas como el informe de clasificación y el área bajo la curva ROC (AUC).
