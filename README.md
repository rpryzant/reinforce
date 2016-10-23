## Final project for cs221 - AI

### project description
* algo: Q Learning with function approximation? policy gradients?
* off policy learning from multiple exploratory policies (one per thread)
* off policy learning on previous/stale policies? (bootstrap on policy)
* use q learning for off policy learning: next action is chosen using behavior policy, also consider altertative successor action drawn from foreign policy. Bootstrap on this alternative successor action. Target policy should be **greedy** w.r.t Q, behavior policy should be **epsilon-greedy** with respect to Q
* different kinds of function approximators? (neural net, logistic regression, etc)
  * td(0), td(lambda), backward view td, mc
* state vector should include notion of actions!!!
* radial basis, course coding for function approximators?
* do TD, find best lambda value?
* use experience replay - store experience (state, value) pairs
  * **LECTURE 6 1:18!!** 
  * experience replay is good because it decorrelates trajectories
  * use old params for target calculation for more stable update
    * experiment with smooth interpolation between old, fresh weights? (papers swap weights every once in a while)


## Questions for percy
* we should be doing model free learning, right?
* what to use as states? need to capture 
  * paddle pos
  * ball pos & movement
  * remaining brick positions
  * discretize state space for ball pos + movement? (called *tile coding*)
* will q-learning be enough? big(ish) temporal difference between good action (hitting ball w right angle) and reward (points)



## inspiration/resources
* rl textbook: https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html
* reinforcement learning lectures/articles/notes
  * http://cs229.stanford.edu/notes/cs229-notes12.pdf
  * https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
  * https://www.nervanasys.com/deep-reinforcement-learning-with-neon/
  * stanford
    * https://www.youtube.com/watch?v=RtxI449ZjSc
    * https://www.youtube.com/watch?v=LKdFTsM3hl4
    * https://www.youtube.com/watch?v=-ff6l5D8-j8
  * berkeley
    * https://www.youtube.com/watch?v=IXuHxkpO5E8
    * https://www.youtube.com/watch?v=yNeSFbE1jdY
  * deepmind course
    * http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
* a group last year did reinforcement learning
  * http://web.stanford.edu/class/cs221/restricted/projects/piq93/final.pdf
  * http://web.stanford.edu/class/cs221/restricted/projects/piq93/poster.pdf
* various ML groups have done reinforcement learning
  * http://cs229.stanford.edu/proj2015/362_report.pdf
  * http://cs229.stanford.edu/proj2015/362_poster.pdf
  * http://cs229.stanford.edu/proj2014/Shuhui%20Qu,%20Tian%20Tan,Zhihao%20Zheng,%20Reinforcement%20Learning%20With%20Deeping%20Learning%20in%20Pacman.pdf
  * http://cs229.stanford.edu/proj2013/CamDembiaIsraeli-RLBicycle.pdf
  * http://cs229.stanford.edu/proj2012/BodoiaPuranik-ApplyingReinforcementLearningToCompetitiveTetris.pdf
  * http://cs229.stanford.edu/proj2012/LiaoYiYang-RLtoPlayMario.pdf
* Google deepmind did breakout
  * https://arxiv.org/pdf/1312.5602.pdf
