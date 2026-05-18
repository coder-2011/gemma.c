# Worklog

This file is a work log of how I went about creating the megakernel
Note that I have two work logs. experiments/EXPERMIENTS.md is fully agent maintained, annoying to read, and verbose.
This one is human maintained, and lightly AI formatted, as well as more relevant to read.

- The general approach for my work through the mega kernel is to start by creating an unfused implementation every kernel, but writing ways that is easy to maintain and fuse later. This is largely to get a feel for what the model is like, because Gemma 4 is a unique, feature-dense architecture and A6000 is a GPU I haven't created many kernels for before.


- A design decision I'm taking is writing custom mat model kernels and trying to rely as little on CU DNN as possible such that I can squeeze as much performance as possible. This is really meant to be as optimal and easy to use as possible, and that requires me to work my way through the stack.

- As of now I'm working on custom matmul and GEMV kernels for the fixed projection shapes in the Gemma 4 31B dense path. 

- Matmul kernels have been finished, code quality has to be improved, and I think there's a couple of issues with cache hints. Unfortunately, I'm not running on bare metal GPUs, so I cannot use _nsys_, but hopefully if I get my hands on some baremetal A6000s, I can run nsys and check for issues. All the bells and whistles are implemented.

- For the RMS-norm, I pretty much ported llm.c's RMSNorm, and worked off that. it was really cleanly implemented so there is no need to rewrite it. Ofc, adjustments have to be made since we use RMS instead of LayerNorm.
