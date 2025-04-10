# ASat Monomer Classifier
Using Long Short-Term Memory (LSTM) network for classifying monomers in alpha satellites


# Characterizing centromeric monomers at the read level using LSTM

The smallest repeat units in centromeric alpha satellites (ASats) are called monomers. Monomers
are ∼171 bases long and are usually organized into Higher Order Repeats (HORs). Each re-
peat unit in HORs is a concatenation of multiple monomers. Although the repeated monomers
within each HOR unit can be diverged (50%-90% identical) the HOR units are highly homoge-
nous (95%-100% identical).

Since the HOR units are highly similar it is challenging for assemblers to distinguish
true read overlaps from the spurious ones. That is why even the state-of-the-art assemblers
produce fragmented erroneous HORs. CentroFlye is an assembler designed specifically for cen-
tromeres and it takes advantage of the monomeric composition of centromeric long reads to
avoid false overlaps. CentroFlye uses StringDecomposer to decompose long ONT reads
into monomers. Then it constructs De-Bruijn graphs iteratively using monomeric representa-
tions instead of canonical bases. For running StringDecomposer we need a set of monomers,
which is used for alignment and the further decomposition of the given sequences (ONT long
reads in this case). It also provides a classification of the detected monomers, which is depen-
dent on the monomers initially given to StringDecomposer. I will propose a Recurrent Neural
Network (RNN)-based model for converting reads bases into monomer units with no need for
a base set of monomers. I will then suggest a clustering scheme for monomers using LSTM
encoder, which has the potential for increasing the resolution of monomer classification. The
output of clustering can then be used for finding overlaps between the centromeric reads more
effectively.


## 1 Methods
### 1.1 Data preparation

The first step in detecting and classifying monomers is creating a data set, which can be used for
training and testing the classifier model. Since the goal is analyzing monomers at the level of the
reads this data set should contain a set of reads annotated with centromeric monomers. One way
to create such a data set is by aligning reads to the CHM13-v1.1 assembly whose monomers are
annotated in detail by the T2T consortium. To avoid mixing up read errors with true variants
the reads should be from the CHM13 cell line. After aligning reads we can project the monomer
intervals and their classes to the read coordinates. It is also possible to use annotation tools like
StringDecomposer to annotate each read individually without looking at reference. One major
benefit of the projection-based approach is that read errors cannot misguide the annotation
because the original annotation was performed at the level of a high quality assembly. On the
other hand one drawback is that read mismappings will lead to false projections. To alleviate
this issue we can include only reliable mappings (This is not investigated yet). I write a python
program, project blocks.py, for doing the projection (Already introduced in Aim1). It does that
by iterating over the CIGAR strings and associating read bases with assembly bases.
The training read set consisted of 16,000 reads randomly selected from the CHM13
ONT reads base-called with Guppy3.6.0. A completely separate set of 4,000 reads were randomly
selected to be used as the test set and evaluating the model after training.

### 1.2 Classifying read bases
Monomer detection can be stated as a classification problem by categorizing bases into three
groups; non-monomeric bases, the starting base of each monomer, and internal monomeric bases.
The second group which points to start locations is necessary since in Higher Order Repeats
it is common to have monomers beside each other without any interruptions in between. The
third group which points to internal bases is also essential since the monomer lengths are not
fixed although they usually fluctuate around 171 bp. If a model could classify the bases of
a sequence into these three groups, we can then easily extract the monomer intervals in that
sequence. ASat monomers can be classified based on their sequence content into different supra
chromosomal families (SFs) and each SF has its own SF-specific monomer classes provided by
the T2T consortium. Instead of having two models; one for identifying monomer boundaries
and one for classifying them into monomer classes we can merge them into one model. We can
aggregate the SF-specific classes with the non-monomeric group and the starting-base group
described earlier. It will provide a set of groups for detecting monomer boundaries and their
SF-specific classes through a single classifier model. The monomer classes that are not within
HOR arrays are grouped into one class named “other”. Because of lack of data for some rare
classes the model will have difficulty in learning their patterns so I decreased the classifying
resolution for such classes. The monomer classes with low frequency (less than 0.1%) among
the HOR monomers were merged into the “other” group. The low frequency classes included
D4, D6 and R2.

### 1.3 Model architecture
The LSTM architecture was used for learning monomer classes. The size of each hidden and
cell state was selected to be 64. The model was bidirectional which means having two separate
directions for propagating hidden and cell states; one from the beginning of the sequence to end
and the other one in the reverse direction. This way at each locus we can obtain information
from both sides. Furthermore each direction had two layers of LSTM units stacked on top of
each other. The final output at each locus will consist of two hidden vectors; one from forward
direction and one from backward. Consequently the output of LSTM at each locus will be a
vector of length 128. This vector will then be fed to one final layer, which transforms it into a
vector of the same size as the number of monomer classes. The activation function of soft-max
will then be applied to have a vector of probabilities. The truth monomer class should ideally
have the highest probability across all 18 classes. The cross entropy loss function was used
for comparing the output of the model against the true classes. The model parameters were
updated with Adam algorithm.


### 2 Preliminary Results
#### 2.1 Training classifier
For training the LSTM-based classifier, I split reads into non-overlapping chunks of length 500.
The chunks were then shuffled and grouped into batches. Each batch contained 64 random
chunks. The model parameters were updated after feeding each batch to the model and com-
puting the batch loss using cross entropy loss function. One epoch means training using the
whole training set and I stopped the training process after 4 epochs since the loss value remained
around 0.1 and did not change after that. I calculated the loss value averaged for every 100
batches. Figure 5.1 shows how it has changed during training.

<img src="https://github.com/mobinasri/monomer_classifier/blob/main/loss.png?raw=true" alt="Alt text" width="500"/>

**Figure 1: Changes in the cross entropy loss during training the LSTM-based classifier using 16,000 ONT reads**


#### 2.2 Evaluating classifier
After training was stopped after 4 epochs the model was validated on the chunks extracted
from the test set. The confusion matrix plotted in Figure 2 compares the model’s prediction
and the true monomer classes. For most of the classes it could classify them with higher than
99% recall and precision. The model had the lowest performance on predicting D3 monomer,
with 59% recall (55% precision). It was∼37% of the time misclassified as D1. This poor
classification is mainly due to the low frequency of D3 sequences in the CHM13 reference so
the training set does not have enough D3 examples to train the model. The start locations of
monomers could be identified by 96% recall rate and 75% precision. Although the recall rate
is high It needs more investigation to understand why the precision is low here. Since these
numbers are calculated at the base level one hypothesis can be the the false positives are close
to the true locations that are already found by the model but it lacks confidence in finding the
exact base. One other hypothesis could be that these start locations are not projected correctly
due to mismapping in the read alignments. Enlarging and diversifying the training dataset,
ignoring unreliable mappings and altering the complexity of the model are some of the possible
solutions for addressing the low accuracy of the model in detecting starting bases and classifying
monomers like D3.

<img src="https://github.com/mobinasri/monomer_classifier/blob/main/recall.png?raw=true" alt="Alt text" width="500"/>
<img src="https://github.com/mobinasri/monomer_classifier/blob/main/precision.png?raw=true" alt="Alt text" width="500"/>

**Figure 2: The confusion matrix obtained by testing the classifier on 4,000 ONT reads. The confusion matrix was once normalized row-wise (top panel) and once column-wise (bottom panel) to calculate recall and precision rates respectively**


## Future Directions

- I could train an LSTM-based classifier for predicting the locations of the HOR monomer
and also their SF-specific classes at the level of the reads. However this model is not
performing as expected for D3 monomers. Increasing the number of training examples
and balancing monomer classes in the training set can potentially solve the problem.
- Another next step will be comparing this model against tools like StringDecomposer[8]
and HumAS-HMMER [2].
- The ONT reads used for the current results were basecalled with Guppy3.6.0 however at
the time of writing this there exist newer Nanopore chemistries and also newer releases of
Guppy. To have an updated model with the ability of removing read errors it should be
trained on the latest ONT reads.
- The current results show that LSTM is able to learn monomer’s content and distinguish
different types of monomers from each other. I like to explore the numerical representa-
tions produced by LSTM. To be more specific an Encoder-Decoder LSTM can be trained
to receive a monomer sequence in encoder and reconstruct it in the decoder. After training
we can eliminate the decoder part and use encoder to generate a numerical representation
of any given monomer. This representation can be the concatenation of the hidden and
cell states produced by the last encoder unit. The size of this encoded vector should be set
while specifying the model. It can then be used for clustering monomers based on their
content. This clustering is expected to be in concordance with the supra chromosomal
families (SFs) and also SF-specific groups. Furthermore by increasing the number of clus-
ters (for example by increasing k in k-means clustering) we should be able to increase the
resolution of clustering. Increasing the resolution of monomer classes will facilitate finding
overlaps between reads after transforming them into their monomeric representations. An
overview of this idea is shown in Figure 3.
