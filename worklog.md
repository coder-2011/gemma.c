# Worklog

This file is a work log of how I went about creating the megakernel
Note that I have two work logs. experiments/EXPERMIENTS.md is fully agent maintained, annoying to read, and verbose.
This one is human maintained, and lightly AI formatted, as well as more relevant to read.

- The general approach for my work through the mega kernel is to start by creating an unfused implementation every kernel, but writing ways that is easy to maintain and fuse later. This is largely to get a feel for what the model is like, because Gemma 4 is a unique, feature-dense architecture and A6000 is a GPU I haven't created many kernels for before.


- A design decision I'm taking is writing custom mat model kernels and trying to rely as little on CU DNN as possible such that I can squeeze as much performance as possible. This is really meant to be as optimal and easy to use as possible, and that requires me to work my way through the stack.

- As if now I'm working on custom matmul and GEMV kernels. Specific ones for diff matmuls we have to do. Outperformed cuDNN by ~xx percent on avg [AGENTS FILL IN THIS INFORMATION IF YOU HAVE IT]
