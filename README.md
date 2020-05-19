
# several_model_in_tf2
test code for saving, loading, optimizing several model at once

### Dependences
tensorflow v2.0 or greater  
and   
numpy any version

### How to test
```
python some_model.py
or
python some_modelSequential.py
```
### What does it do inside the code.
-----------

model1 trains y = 1 (x)  
model2 trains y = 2 (x)  
model3 trains y = model1(x) + model2(x)

-----------
