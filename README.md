# droppedaneuralnet

Solution to Jane Street's "I dropped a neural net" ([here](https://huggingface.co/spaces/jane-street/droppedaneuralnet)). 

Pretty much implements [this tweet](https://x.com/0xdjma/status/2020377562267152582):
1. Last layer is easy to find since it's the only one with a different shape.
1. Form initial blocks (pairs of input/output pieces) by cosine similarity.
1. Form initial permuation by $L_2$ norm.
1. Brute force search by swapping blocks until network preds match ~identically. Usually ~10k iterations will find the correct permuation. 

Other things I tried:
1. Create blocks (pairs of input/output pieces) by by minimizing RMSE. Finds ~60% of the correct blocks but requires finer-grained search in step (4) above since you also need to search the blocks.
1. Greedy search, etiher front-to-back or back-to-front.
1. Huge, parallel search. This is also what gpt-5.3-codex-medium will try to do if you just give it the problem description. This probably works but felt too brute force. 
1. Looking at weight and activation statistics, norms, etc. There was not much correlation and mostly didn't work.

Things I didn't try because they seemed too annoying / unpromisng:
1. [DARTS](https://arxiv.org/abs/1806.09055)-style architecture search -- search space may be too large.
2. Had an idea of starting from the final layer, differentiating into the inputs to find the input activations which lead to the lowest loss, then trying to find the block with output activations which most closely match these learned ones. Kind of half-baked, not sure it would work.

All code was written entirely by clanker, specifically gpt-5.3-codex.