# TODO

*Agent maintained, quite messy. They never listen to me when I tell them to keep it clean...*

- Rewrite the main megakernel path fully, as it is quite broken right now. Just scrap it, and rewrite.
- Confirm and audit all indiviual path benchmarks, and clean them up. I bet a sophisticated agent loop can do this in one shot, but I am not entirely sure. It might take some more manual work
- Get rid of the random slop that is in this repo, like the SASS files, and make sure all important things to gitignore and ignored.
- Audit gemma4.cpp( and .h) bc they seem really redundant right now. The stuff in them can prob js go into the utils.
- We need to run a /goal to tune basic hyperparams for quasi-kernels, I think we can squeeze another 10-20% more efficency just off that alone.
- Normal optimization of quasi-kernels
- It is important to look at the whole path from a more holistic perspective, but still very thoroughly.
