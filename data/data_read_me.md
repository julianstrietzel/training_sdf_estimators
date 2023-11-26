In the subfolders of data place the data for each dataset.

For example, for SHREC place the data in ```data/shrec```. The data should be in xyz format like in the bacon codebase
and not yet normalized.

The data should be in the following format:

```
5.894578 11.788401 27.283239 -0.329976 -2.845177 5.575466 
-53.325111 67.104362 -57.450130 -2.107734 -1.572878 -5.502226 
3.750489 16.505402 29.454020 3.606345 1.496845 4.914672 
...
```

Where each line is a vertex and the first three numbers are the x, y, z coordinates of the vertex and the last three
numbers are the normal of the vertex.
