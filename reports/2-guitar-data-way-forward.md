I started out with the objective of automatic guitar music transcription.

The goal for my first month was to understand the landscape of this
problem and validate feasibility by training existing models on existing
datasets and making this work for "clean guitar music".
I have been able to do this successfully!

For the first month, I just wanted to understand and validate the technical
feasibility. Now, I have the following goals:
1. User research: there are two major user profiles - beginners (like
2. Competitive ree
3. Think about how

The answers to these will determine how I proceed with this project.

At the beginning, this was all an idea and I had no background in signal
processing at all. I relied on my machine learning skills to

Now, I have a working prototype! Here is a sample output that it produces:


And it works (okay-ish) on audio that I took from the first 1 minute of
this video!

With this, I have validated the feasibility of
There is a lot of room for improvement in the model right now:
1. Need to predict the "offsets" (where the note ends) so that it is
    possible to actually play the transcribed music back
1. This is a simplistic model - there is


Most of these problems have actually been solved in [Onsets and Frames]()
work from Google Magenta. They even have an [amazing demo which works
for piano music transcription]().







## Exploring the existing solutions

I spent a fair amount of time this week trying to explore the existing
solutions that

### Traditional software

There are a whole bunch of notation software like MuseScore, Guitar Pro,
etc. Most of these are mainly tools for writing down sheet music

###
There's actually a fair amount of work in Automatic Music Transcription
using Machine Learning [1](), [2](), [3](), etc. The most notable of
these is Google Magenta's piano music transcription. They have an
[amazing demo as well]()!

I also found []() which is quite similar (but better) to the baseline
model that I created! It also works majorly for piano music (was trained
on the MAPS dataset).


## The way forward




High-level:
1. Make this work on real videos and songs (still mono-phonic)
1. Bridge the gap between data from real videos and artificial datasets
1. I have spent very little time on the underly

What is the way forward (High-level?
