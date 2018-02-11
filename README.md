# FMLite
Pure Python Implementation of Facorization Machine.

## Author
- moriaki3193

## Setup
```
$ pip install FMLite
```

## Usage
You can easily use this package like below.

```Python
from fmlite import FMLite

model = FMLite(k=10, n_epochs=20) # and some other kwargs...
model.fit(X_train, y_train)
model.predict(X_test)
```

More detailed examples are in `samples` directory.

## Requirements
- Python 3.6.0 or newer
