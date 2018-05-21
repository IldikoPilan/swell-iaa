import json
import codecs
import os
import nltk
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import masi_distance, jaccard_distance



def sum_results(data_folder):
    """ Collect in one object annotations from different annotators and texts.
    Returns a dictionary with the following structure:
    {annotator1 : {text1 : {edge1 : label1, edge2 :label2,...}...}}
    """
    summed_results = {}
    for annotator in os.listdir(data_folder):
        if "." not in annotator: 
            summed_results[annotator] = {}
            annot_file = os.path.join(data_folder, annotator + "/state")
            with codecs.open(annot_file) as f:
                decoded_annot = json.load(f)
                for k,v in decoded_annot.items():
                    if k == "graphs":
                        for text,text_val in v.items():
                            if text != "examples":
                                edges = text_val["graph"]["now"]["edges"] #source
                                annot_edges = {}
                                for edge, info in edges.items():
                                    annotat_val = info["labels"]

                                    # Consider only source tokens
                                    source_edges = "-".join([edge_id for edge_id in info["ids"] if "s" in edge_id])
                                    
                                    # Consider all edges
                                    #if not annotat_val:     # no annotation -> correct
                                    #    annotat_val = ["CORR"]
                                                                        
                                    if annotat_val: 
                                        # Consider ONLY annotated edges
                                        annot_edges[source_edges] = annotat_val

                                if annot_edges:
                                    summed_results[annotator][text] = annot_edges
    return summed_results

data_folder = "/Users/xpilil/Documents/DEVELOPMENT/SWELL-RJ/iaa/data/"
summed_results =  sum_results(data_folder)

def anonymize_annotator(current_annot_ix):
    """ Returns a code instead of the name of an annotator 
    in the format 'cX' where X is a consequtive number representing
    the order of annotators, the count starting from 1. 
    """
    return "c"+str(current_annot_ix + 1)

def create_iaa_data(summed_results, text, annotators, dummy_label, flexible, add_missing=True):
    """ Returns a list of (annotator, edge, label) usable as input for an AnnotationTask. 
    @ flexible: 	bool (True = project multi-token annotations to single tokens)
    @ add_missing:  bool (True = pad missing annotations with a dummy label, V1 in presentation)
    """
    iaa_data = []
    for current_annot_ix,annotator in enumerate(annotators):
        annotations = summed_results[annotator][text]
        anonym_annot = anonymize_annotator(current_annot_ix)
        for edge, labels in annotations.items():
            if flexible:
                split_edge = edge.split("-")
                for single_edge in split_edge:
                    iaa_data.append((anonym_annot, single_edge, frozenset(labels)))
            else:
                iaa_data.append((anonym_annot, edge, frozenset(labels)))


        # Compare to other annotators and add any missing edges with 
        # empty labels as value.
        if add_missing:
            edges_current_annot = [e for (c,e,l) in iaa_data if c == anonym_annot]
            for j in range(len(annotators)):
                if j != current_annot_ix:
                    other_annotator = annotators[j]
                    for edge2,label2 in summed_results[other_annotator][text].items():
                        if flexible:
                            split_edge2 = edge2.split("-")
                            for single_edge2 in split_edge2:
                                if single_edge2 not in edges_current_annot:
                                    if (anonym_annot,edge2,dummy_label) not in iaa_data: # to avoid duplicates
                                        iaa_data.append((anonym_annot, single_edge2, dummy_label))
                                    
                        else:
                            # Disagreemnts on edge (and consequently also on label)
                            if edge2 not in summed_results[annotator][text]:
                                if (anonym_annot,edge2,dummy_label) not in iaa_data: 	 # to avoid duplicates
                                    iaa_data.append((anonym_annot, edge2, dummy_label))
                       
    return iaa_data

#text = "text3"
#annotators = ["beata", "elena", "julia"] # "text3"

text = "text6" 
annotators = ["beata", "julia","mats"] # "text6"

dummy_label = frozenset(["CORR"])
flexible = False
add_missing = False		# True = V1, False = V2
iaa_data = create_iaa_data(summed_results, text, annotators, dummy_label, flexible, add_missing)

#print iaa_data[:3]

task = AnnotationTask(data=iaa_data,distance=jaccard_distance)

print "**** Inter-annotator agreement for", text, "****"
print "Avg agreement:\t\t\t\t", round(task.avg_Ao(),3)    		# Average observed agreement across all coders and items.
print "Fleiss (multi_kappa):\t\t", round(task.multi_kappa(),3)  # (Davies and Fleiss 1982)
print "Krippendorff's alpha:\t\t", round(task.alpha(),3) 		# (Krippendorff 1980)