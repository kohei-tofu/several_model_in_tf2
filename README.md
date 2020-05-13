<script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [["\\(","\\)"] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>



# several_model_in_tf2
test code for saving, loading, optimizing several model at once

### Dependences
tensorflow v2.0 or greater
numpy any version

### How tot test
```
python some_model.py
or
python some_modelSequential.py
```
### What does it do inside the code.
model1 trains $$ y = 1 \{x} $$
model2 trains $$ y = 2 \{x} $$
model3 trains $$ y = model1{x} + model2{x} $$


