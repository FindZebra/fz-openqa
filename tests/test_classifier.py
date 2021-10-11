from unittest import TestCase

import torch

from fz_openqa.datamodules.pipes.relevance import MetaMapMatch, SciSpacyMatch, ExactMatch

class TestClassifier(TestCase):
    def setUp(self) -> None:
        self.question = [
            "A 59-year-old overweight woman presents to the urgent care clinic with the complaint of severe abdominal pain for the past 2 hours. She also complains of a dull pain in her back with nausea and vomiting several times. Her pain has no relation with food. Her past medical history is significant for recurrent abdominal pain due to cholelithiasis. Her father died at the age of 60 with some form of abdominal cancer. Her temperature is 37\u00b0C (98.6\u00b0F), respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. Physical exam is unremarkable. However, a CT scan of the abdomen shows a calcified mass near her gallbladder. Which of the following diagnoses should be excluded first in this patient?",
            "A 67-year-old man who was diagnosed with arthritis 16 years ago presents with right knee swelling and pain. His left knee was swollen a few weeks ago, but now with both joints affected, he has difficulty walking and feels frustrated. He also has back pain which makes it extremely difficult to move around and be active during the day. He says his pain significantly improves with rest. He also suffers from dandruff for which he uses special shampoos. Physical examination is notable for pitting of his nails. Which of the following is the most likely diagnosis?"
            ]
        self.pos_answer = [
            {'answer.target': 1, 'answer.text': ["Acute cholecystitis", "Gallbladder cancer", "Choledocholithiasis","Pancreatitis"], 'answer.cui':['C0235782'], 'answer.synonyms':[]},
            ]
        self.neg_answer = [
            {'answer.target': 0, 'answer.text': ["Psoriatic arthritis","Arthritis mutilans","Rheumatoid arthritis","Mixed connective tissue disease"], 'answer.cui':['C0003872'], 'answer.synonyms':[]}
            ]
        self.pos_document = [
            {'document.text': "Carcinoma of the gallbladder (GBC) is the most common and aggressive form of biliary tract cancer (BTC; see this term) usually arising in the fundus of the gallbladder, rapidly metastasizing to lymph nodes and distant sites. Epidemiology Annual incidence rates vary from 1/100,000 to 1/ 4,350 between different ethnic groups and geographical regions. It is rare in developed Western countries but has a high incidence in Japan (1/19,000), northern India, Chile and certain regions of Eastern Europe. Clinical description GBC is a rare neoplasm occurring more often in females (3-4:1 female to male ratio) with an average age of onset of 65 years. Most patients are asymptomatic until the disease is advanced but presenting symptoms include abdominal pain (usually in the upper right quadrant), nausea, vomiting, jaundice, anorexia and weight loss. Gallstones are often present in patients with GBC. GBC is extremely aggressive and invasion of the lymph nodes, liver and other organs occurs rapidly in many cases. Etiology The exact etiology is unknown. Genetic susceptibility elicited by chronic inflammation of the gallbladder leading to dysplasia and malignant change is one possibility. Risk factors associated with GBC include a history of gallstones, cholelithiasis, porcelain gallbladder, bacterial infections, high caloric diet"},
            {'document.text': "the fingers, nails, and skin. Sausage-like swelling in the fingers or toes, known as dactylitis, may occur. Psoriasis can also cause changes to the nails, such as pitting or separation from the nail bed, onycholysis, hyperkeratosis under the nails, and horizontal ridging. Psoriasis classically presents with scaly skin lesions, which are most commonly seen over extensor surfaces such as the scalp, natal cleft and umbilicus. In psoriatic arthritis, pain can occur in the area of the sacrum (the lower back, above the tailbone), as a result of sacroiliitis or spondylitis, which is present in 40% of cases. Pain can occur in and around the feet and ankles, especially enthesitis in the Achilles tendon (inflammation of the Achilles tendon where it inserts into the bone) or plantar fasciitis in the sole of the foot. Along with the above-noted pain and inflammation, there is extreme exhaustion that does not go away with adequate rest. The exhaustion may last for days or weeks without abatement. Psoriatic arthritis may remain mild or may progress to more destructive joint disease. Periods of active disease, or flares, will typically alternate with periods of remission. In severe forms, psoriatic arthritis may progress to arthritis mutilans which on"},
            ]
        self.neg_document = [
            {'document.text': "CholecystitisAcute cholecystitis as seen on CT. Note the fat stranding around the enlarged gallbladder.SpecialtyGeneral surgery, gastroenterologySymptomsRight upper abdominal pain, nausea, vomiting, feverDurationShort term or long termCausesGallstones, severe illnessRisk factorsBirth control pills, pregnancy, family history, obesity, diabetes, liver disease, rapid weight lossDiagnostic methodAbdominal ultrasoundDifferential diagnosisHepatitis, peptic ulcer disease, pancreatitis, pneumonia, anginaTreatmentGallbladder removal surgery, gallbladder drainagePrognosisGenerally good with treatment Cholecystitis is inflammation of the gallbladder. Symptoms include right upper abdominal pain, nausea, vomiting, and occasionally fever. Often gallbladder attacks (biliary colic) precede acute cholecystitis. The pain lasts longer in cholecystitis than in a typical gallbladder attack. Without appropriate treatment, recurrent episodes of cholecystitis are common. Complications of acute cholecystitis include gallstone pancreatitis, common bile duct stones, or inflammation of the common bile duct. More than 90% of the time acute cholecystitis is from blockage of the cystic duct by a gallstone. Risk factors for gallstones include birth control pills, pregnancy, a family history of gallstones, obesity, diabetes, liver disease, or rapid weight loss. Occasionally, acute cholecystitis occurs as a result of vasculitis or chemotherapy, or during recovery from major trauma or burns. Cholecystitis is suspected based on symptoms and laboratory testing. Abdominal ultrasound is then typically used to confirm the diagnosis. Treatment"},
            {'document.text': "commonly involved joints are the two near the ends of the fingers and the joint at the base of the thumbs; the knee and hip joints; and the joints of the neck and lower back. Joints on one side of the body are often more affected than those on the other. The symptoms can interfere with work and normal daily activities. Unlike some other types of arthritis, only the joints, not internal organs, are affected. Causes include previous joint injury, abnormal joint or limb development, and inherited factors. Risk is greater in those who are overweight, have legs of different lengths, or have jobs that result in high levels of joint stress. Osteoarthritis is believed to be caused by mechanical stress on the joint and low grade inflammatory processes. It develops as cartilage is lost and the underlying bone becomes affected. As pain may make it difficult to exercise, muscle loss may occur. Diagnosis is typically based on signs and symptoms, with medical imaging and other tests used to support or rule out other problems. In contrast to rheumatoid arthritis, in osteoarthritis the joints do not become hot or red. Treatment includes exercise, decreasing joint stress such as by rest"}
            ]
        self.model_name = "en_core_sci_lg"

        self.ExactMatch = ExactMatch()
        self.SciSpacyMatch = SciSpacyMatch()
        self.MetaMapMatch = MetaMapMatch()

    def test_exact_match(self):
        self.assertFalse(self.ExactMatch.classify(answer = self.pos_answer[0], document = self.pos_document[0]))
        #self.assertFalse(ExactMatch.classify(answer = self.neg_answer, document = self.neg_document))

    def test_metamap_match(self):
        self.assertTrue(self.MetaMapMatch.classify(answer = self.pos_answer[0], document = self.pos_document[0]))

    def test_scispacy_match(self):
        self.assertTrue(self.SciSpacyMatch.classify(answer = self.pos_answer[0], document = self.pos_document[0]))
