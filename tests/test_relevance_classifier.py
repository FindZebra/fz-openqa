from unittest import TestCase
from copy import copy

import torch
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes.relevance import MetaMapMatch, ScispaCyMatch, \
    ExactMatch, Pair, find_one

b0 = {'question.text': "What is the symptoms of post polio syndrome?",
      "answer.target": 0, "answer.text": ["Post polio syndrome (PPS)"],
    "document.text": ["Post polio syndrome is a condition that affects polio survivors years after recovery from the initial polio illness. Symptoms and severity vary among affected people and may include muscle weakness and a gradual decrease in the size of muscles (atrophy); muscle and joint pain; fatigue;difficulty with gait; respiratory problems; and/or swallowing problems.\xa0Only a polio survivor can develop PPS. While polio is a contagious disease, PPS is not. The exact cause of PPS years after the first episode of polio is unclear, although several theories have been proposed. Treatment focuses on reducing symptoms and improving quality of life."]}

b1 = {'question.text': "A pulmonary autopsy specimen from a 58-year-old woman who died of acute hypoxic respiratory failure was examined. She had recently undergone surgery for a fractured femur 3 months ago. Initial hospital course was uncomplicated, and she was discharged to a rehab facility in good health. Shortly after discharge home from rehab, she developed sudden shortness of breath and had cardiac arrest. Resuscitation was unsuccessful. On histological examination of lung tissue, fibrous connective tissue around the lumen of the pulmonary artery is observed. Which of the following is the most likely pathogenesis for the present findings?",
      "answer.target": 0, "answer.text": ["Thromboembolism"], "answer.synonyms":[],
      "document.text": ["secretion. The Eicosanoids: Prostaglandins, Thromboxanes, Leukotrienes, & Related Compounds John Hwa, MD, PhD, & Kathleen Martin, PhD* pulmonary pressures, and right ventricular enlargement. Cardiac catheterization confirmed the severely elevated pulmonary pressures. She was commenced on appropri-ate therapies. Which of the eicosanoid agonists have been demonstrated to reduce both morbidity and mortality in patients with such a diagnosis? What are the modes of action? A 40-year-old woman presented to her doctor with a 6-month history of increasing shortness of breath. This was associated with poor appetite and ankle swell-ing. On physical examination, she had elevated jugular venous distention, a soft tricuspid regurgitation murmur, clear lungs, and mild peripheral edema. An echo"]}

b2 = {'question.text': "A 59-year-old overweight woman presents to the urgent care clinic with the complaint of severe abdominal pain for the past 2 hours. She also complains of a dull pain in her back with nausea and vomiting several times. Her pain has no relation with food. Her past medical history is significant for recurrent abdominal pain due to cholelithiasis. Her father died at the age of 60 with some form of abdominal cancer. Her temperature is 37\u00b0C (98.6\u00b0F), respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. Physical exam is unremarkable. However, a CT scan of the abdomen shows a calcified mass near her gallbladder. Which of the following diagnoses should be excluded first in this patient?",
      'answer.target': 1, 'answer.text': ["Acute cholecystitis", "Gallbladder cancer", "Choledocholithiasis","Pancreatitis"], 'answer.cui':['C0235782'], "answer.synonyms":["Gallbladder Carcinoma","Malignant neoplasm of gallbladder"],
      "document.text": ["Carcinoma of the gallbladder (GBC) is the most common and aggressive form of biliary tract cancer (BTC; see this term) usually arising in the fundus of the gallbladder, rapidly metastasizing to lymph nodes and distant sites. Epidemiology Annual incidence rates vary from 1/100,000 to 1/ 4,350 between different ethnic groups and geographical regions. It is rare in developed Western countries but has a high incidence in Japan (1/19,000), northern India, Chile and certain regions of Eastern Europe. Clinical description GBC is a rare neoplasm occurring more often in females (3-4:1 female to male ratio) with an average age of onset of 65 years. Most patients are asymptomatic until the disease is advanced but presenting symptoms include abdominal pain (usually in the upper right quadrant), nausea, vomiting, jaundice, anorexia and weight loss. Gallstones are often present in patients with GBC. GBC is extremely aggressive and invasion of the lymph nodes, liver and other organs occurs rapidly in many cases. Etiology The exact etiology is unknown. Genetic susceptibility elicited by chronic inflammation of the gallbladder leading to dysplasia and malignant change is one possibility. Risk factors associated with GBC include a history of gallstones, cholelithiasis, porcelain gallbladder, bacterial infections, high caloric diet"]}

b3 = {'question.text': "A 67-year-old man who was diagnosed with arthritis 16 years ago presents with right knee swelling and pain. His left knee was swollen a few weeks ago, but now with both joints affected, he has difficulty walking and feels frustrated. He also has back pain which makes it extremely difficult to move around and be active during the day. He says his pain significantly improves with rest. He also suffers from dandruff for which he uses special shampoos. Physical examination is notable for pitting of his nails. Which of the following is the most likely diagnosis?",
      'answer.target': 0, 'answer.text': ["Psoriatic arthritis","Arthritis mutilans","Rheumatoid arthritis","Mixed connective tissue disease"], 'answer.cui':['C0003872'], "answer.synonyms":["Arthritis, Psoriatic"],
      "document.text": ["the fingers, nails, and skin. Sausage-like swelling in the fingers or toes, known as dactylitis, may occur. Psoriasis can also cause changes to the nails, such as pitting or separation from the nail bed, onycholysis, hyperkeratosis under the nails, and horizontal ridging. Psoriasis classically presents with scaly skin lesions, which are most commonly seen over extensor surfaces such as the scalp, natal cleft and umbilicus. In psoriatic arthritis, pain can occur in the area of the sacrum (the lower back, above the tailbone), as a result of sacroiliitis or spondylitis, which is present in 40% of cases. Pain can occur in and around the feet and ankles, especially enthesitis in the Achilles tendon (inflammation of the Achilles tendon where it inserts into the bone) or plantar fasciitis in the sole of the foot. Along with the above-noted pain and inflammation, there is extreme exhaustion that does not go away with adequate rest. The exhaustion may last for days or weeks without abatement. Psoriatic arthritis may remain mild or may progress to more destructive joint disease. Periods of active disease, or flares, will typically alternate with periods of remission. In severe forms, psoriatic arthritis may progress to arthritis mutilans which on"]}


class TestRelevanceClassifier(TestCase):
    """test the RelevanceClassifier"""
    def setUp(self) -> None:
        exs = [b0, b1, b2, b3]
        self.batch = Collate(keys=None)(exs)

    def test_exact_match(self):
        classifier = ExactMatch()
        output = classifier(copy(self.batch))
        # test that exact match matches the document b3 to the answer
        self.assertTrue(output['document.is_positive'][3])
        # test that exact match fails when answer is not contained in document
        self.assertFalse(output['document.is_positive'][0])


    def test_scispacy_match(self):
        classifier = ScispaCyMatch()
        output = classifier(copy(self.batch))
        # test that scispacy match finds the answer in the document using filtering
        self.assertTrue(output['document.is_positive'][3])
        self.assertTrue(output['document.is_positive'][2])
        # test that scispacy is not always true
        self.assertFalse(output['document.is_positive'][1])

class TestFindOne(TestCase):
    """Test the function find one"""
    def setUp(self) -> None:
        self.ops = [None, len, lambda x: -len(x)]

    def test_hello_world(self):
        """test that matching with simple queries and documents"""
        for op in self.ops:
            for doc, queries in [("hello world", ["hello"]),
                                 ("hello world", ["hello", "world"]),
                                 ("hello world", ["world"]),
                                 ("hello world", ["ll"]),]:
                self.assertTrue(find_one(doc, queries, sort_by=op))

    def test_str_case(self):
        """test that matching with simple queries and documents,
        with upper and lowercase inputs."""
        for op in self.ops:
            for doc, queries in [("hello world", ["Hello"]),
                                 ("hello world", ["HELLO"]),
                                 ("hello WOrLd", ["world"]),
                                 ("heLLo world", ["ll"]),]:
                self.assertTrue(find_one(doc, queries, sort_by=op))

    def test_negatives(self):
        """test that find_one returns False where queries are not in the doc."""
        for op in self.ops:
            for doc, queries in [("hello world", ["paris"]),
                                 ("hello world", ["paris", "amsterdam"]),
                                 ("hello world", ["helllo"])]:
                self.assertFalse(find_one(doc, queries, sort_by=op))

    def test_empty_query(self):
        """test the output for empty queries"""
        for op in self.ops:
            for doc, queries in [("hello world", [])]:
                self.assertFalse(find_one(doc, queries, sort_by=op))

    def test_empty_doc(self):
        """test the output for empty docs"""
        for op in self.ops:
            for doc, queries in [("", ["hello", "world"])]:
                self.assertFalse(find_one(doc, queries, sort_by=op))


    #def test_metamap_match(self):
    #    self.assertTrue(self.output.get(self.classifiers[2])['document.is_positive'][3])
