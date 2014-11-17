Inter-frame binary feature coding architecture:
http://home.deib.polimi.it/tagliasa/publications/2014/2014_ICIP_Tagliasacchi_3.pdf

- Open a video
- Extract BRISK local features
- Apply inter-frame coding to sets of local features extracted from contiguous frames
- Compute coding gain


Dependencies:

- OpenCV

The package contains the original BRISK implementation provided by authors:
http://www.asl.ethz.ch/people/lestefan/personal/BRISK