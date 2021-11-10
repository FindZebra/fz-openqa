from copy import copy

import rich
from rich.status import Status

from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes.relevance import ExactMatch
from fz_openqa.datamodules.pipes.relevance import MetaMapMatch
from fz_openqa.datamodules.pipes.relevance import ScispaCyMatch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch

b0 = {
    "question.text": "What is the symptoms of post polio syndrome?",
    "answer.target": 0,
    "answer.text": ["Post polio syndrome (PPS)"],
    "answer.synonyms": [],
    "document.text": [
        "Post polio syndrome is a condition that affects polio survivors years after recovery from the initial polio illness. Symptoms and severity vary among affected people and may include muscle weakness and a gradual decrease in the size of muscles (atrophy); muscle and joint pain; fatigue;difficulty with gait; respiratory problems; and/or swallowing problems.\xa0Only a polio survivor can develop PPS. While polio is a contagious disease, PPS is not. The exact cause of PPS years after the first episode of polio is unclear, although several theories have been proposed. Treatment focuses on reducing symptoms and improving quality of life."  # noqa: E501
    ],
}

b1 = {
    "question.text": "A pulmonary autopsy specimen from a 58-year-old woman who died of acute hypoxic respiratory failure was examined. She had recently undergone surgery for a fractured femur 3 months ago. Initial hospital course was uncomplicated, and she was discharged to a rehab facility in good health. Shortly after discharge home from rehab, she developed sudden shortness of breath and had cardiac arrest. Resuscitation was unsuccessful. On histological examination of lung tissue, fibrous connective tissue around the lumen of the pulmonary artery is observed. Which of the following is the most likely pathogenesis for the present findings?",  # noqa: E501
    "answer.target": 0,
    "answer.text": ["Thromboembolism"],
    "answer.synonyms": [],
    "document.text": [
        "secretion. The Eicosanoids: Prostaglandins, Thromboxanes, Leukotrienes, & Related Compounds John Hwa, MD, PhD, & Kathleen Martin, PhD* pulmonary pressures, and right ventricular enlargement. Cardiac catheterization confirmed the severely elevated pulmonary pressures. She was commenced on appropri-ate therapies. Which of the eicosanoid agonists have been demonstrated to reduce both morbidity and mortality in patients with such a diagnosis? What are the modes of action? A 40-year-old woman presented to her doctor with a 6-month history of increasing shortness of breath. This was associated with poor appetite and ankle swell-ing. On physical examination, she had elevated jugular venous distention, a soft tricuspid regurgitation murmur, clear lungs, and mild peripheral edema. An echo"  # noqa: E501
    ],
}

b2 = {
    "question.text": "Which of the following factors gives the elastin molecule the ability to stretch and recoil?",  # noqa: E501
    "answer.target": 0,
    "answer.text": ["Cross-links between lysine residues"],
    "answer.synonyms": [
        "Lysine",
        "Cross link",
        "Lysine measurement",
    ],
    "document.text": [
        "which are responsible for the elastic properties of the molecule; and alanineand lysine-rich \u03b1-helical segments, which are cross-linked to adjacent molecules by covalent attachment of lysine residues. Each segment is encoded by a separate exon. There is still uncertainty concerning the conformation of elastin molecules in elastic fibers and how the structure of these fibers accounts for their rubberlike properties. However, it seems that parts of the elastin polypeptide chain, like the polymer chains in ordinary rubber, adopt a loose \u201crandom coil\u201d conformation, and it is the random coil nature of the component molecules cross-linked into the elastic fiber network that allows the network to stretch and recoil like a rubber band (Figure 19\u201345).\n\nElastin is the dominant extracellular matrix protein in arteries, comprising 50% of the dry weight of the largest artery\u2014the aorta (see Figure 19\u2013, Deoxypyridinium"  # noqa: E501
    ],
}

b3 = {
    "question.text": "A 59-year-old overweight woman presents to the urgent care clinic with the complaint of severe abdominal pain for the past 2 hours. She also complains of a dull pain in her back with nausea and vomiting several times. Her pain has no relation with food. Her past medical history is significant for recurrent abdominal pain due to cholelithiasis. Her father died at the age of 60 with some form of abdominal cancer. Her temperature is 37\u00b0C (98.6\u00b0F), respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. Physical exam is unremarkable. However, a CT scan of the abdomen shows a calcified mass near her gallbladder. Which of the following diagnoses should be excluded first in this patient?",  # noqa: E501
    "answer.target": 1,
    "answer.text": [
        "Acute cholecystitis",
        "Gallbladder cancer",
        "Choledocholithiasis",
        "Pancreatitis",
    ],
    "answer.cui": ["C0235782"],
    "answer.synonyms": [
        "Gallbladder Carcinoma",
        "Malignant neoplasm of gallbladder",
    ],
    "document.text": [
        "Carcinoma of the gallbladder (GBC) is the most common and aggressive form of biliary tract cancer (BTC; see this term) usually arising in the fundus of the gallbladder, rapidly metastasizing to lymph nodes and distant sites. Epidemiology Annual incidence rates vary from 1/100,000 to 1/ 4,350 between different ethnic groups and geographical regions. It is rare in developed Western countries but has a high incidence in Japan (1/19,000), northern India, Chile and certain regions of Eastern Europe. Clinical description GBC is a rare neoplasm occurring more often in females (3-4:1 female to male ratio) with an average age of onset of 65 years. Most patients are asymptomatic until the disease is advanced but presenting symptoms include abdominal pain (usually in the upper right quadrant), nausea, vomiting, jaundice, anorexia and weight loss. Gallstones are often present in patients with GBC. GBC is extremely aggressive and invasion of the lymph nodes, liver and other organs occurs rapidly in many cases. Etiology The exact etiology is unknown. Genetic susceptibility elicited by chronic inflammation of the gallbladder leading to dysplasia and malignant change is one possibility. Risk factors associated with GBC include a history of gallstones, cholelithiasis, porcelain gallbladder, bacterial infections, high caloric diet"  # noqa: E501
    ],
}

b4 = {
    "question.text": "A 67-year-old man who was diagnosed with arthritis 16 years ago presents with right knee swelling and pain. His left knee was swollen a few weeks ago, but now with both joints affected, he has difficulty walking and feels frustrated. He also has back pain which makes it extremely difficult to move around and be active during the day. He says his pain significantly improves with rest. He also suffers from dandruff for which he uses special shampoos. Physical examination is notable for pitting of his nails. Which of the following is the most likely diagnosis?",  # noqa: E501
    "answer.target": 0,
    "answer.text": [
        "Psoriatic arthritis",
        "Arthritis mutilans",
        "Rheumatoid arthritis",
        "Mixed connective tissue disease",
    ],
    "answer.cui": ["C0003872"],
    "answer.synonyms": ["Arthritis, Psoriatic"],
    "document.text": [
        "the fingers, nails, and skin. Sausage-like swelling in the fingers or toes, known as dactylitis, may occur. Psoriasis can also cause changes to the nails, such as pitting or separation from the nail bed, onycholysis, hyperkeratosis under the nails, and horizontal ridging. Psoriasis classically presents with scaly skin lesions, which are most commonly seen over extensor surfaces such as the scalp, natal cleft and umbilicus. In psoriatic arthritis, pain can occur in the area of the sacrum (the lower back, above the tailbone), as a result of sacroiliitis or spondylitis, which is present in 40% of cases. Pain can occur in and around the feet and ankles, especially enthesitis in the Achilles tendon (inflammation of the Achilles tendon where it inserts into the bone) or plantar fasciitis in the sole of the foot. Along with the above-noted pain and inflammation, there is extreme exhaustion that does not go away with adequate rest. The exhaustion may last for days or weeks without abatement. Psoriatic arthritis may remain mild or may progress to more destructive joint disease. Periods of active disease, or flares, will typically alternate with periods of remission. In severe forms, psoriatic arthritis may progress to arthritis mutilans which on"  # noqa: E501
    ],
}

b5 = {
    "question.text": "a junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician during the case the resident inadvertently cuts a flexor tendon the tendon is repaired without complication the attending tells the resident that the patient will do fine and there is no need to report this minor complication that will not harm the patient as he does not want to make the patient worry unnecessarily he tells the resident to leave this complication out of the operative report which of the following is the correct next action for the resident to take",  # noqa: E501
    "answer.target": 0,
    "answer.text": ["Tell the attending that he cannot fail to disclose this mistake"],
    "answer.synonyms": [
        "error",
        "failed",
        "attending",
        "Attending (action)",
        "attends",
        "failing",
        "To",
        "Attending (provider role)",
        "attended",
        "Tryptophanase",
        "attend",
        "fail",
        "fails",
        "Togo",
    ],
    "document.text": [
        "professional norms of medicine (the Hippocratic oath, respect to patients and colleagues, ethical conduct, personal accountability, empathy, and altruism) are modeled in every personal encounter. It is imperative that all resident and attending surgeons under-stand that the medical students are observing them closely. When resident and attending surgeons model professional behavior, the hidden curriculum becomes a useful tool for professional devel-opment.147-150 This consistent modeling of professional behavior is one necessary component of leadership.During their clinical years, medical students experience both an exponential growth in knowledge and a measurable decline in empathy towards their patients. Initially, medical stu-dents are filled with excitement and wonder during their first patient encounters. The rapid pace of clinical work, acquisition of knowledge, and intense experiences create stress for the stu-dent, both positively and negatively. Scrubbing into the operat-ing room, witn"  # noqa: E501
    ],
}

b6 = {
    "question.text": "a year old man is being treated by his female family medicine physician for chronic depression recently he has been scheduling more frequent office visits he does not report any symptoms or problems with his ssri medication during these visits upon further questioning the patient confesses that he is attracted to her and says you are the only one in the world who understands me the physician also manages his hypertension which of the following is the most appropriate next step in management",  # noqa: E501
    "answer.target": 0,
    "answer.text": ["Ask closed-ended questions and use a chaperone for future visits"],
    "answer.synonyms": [
        "Molecular Chaperones",
        "Future",
        "Patient Visit",
        "Usage",
        "Use - dosing instruction imperative",
        "Visit",
        "Visit Name",
        "utilization qualifier",
        "Does ask questions",
    ],
    "document.text": [
        "is. Similarly, domestic travel may have exposed patients to pathogens that are not normally found in their local environment and therefore may not routinely be considered in the differential diagnosis. For example, a patient who has recently visited California or Martha’s Vineyard may have been exposed to Coccidioides immitis or Francisella tularensis, respectively. Beyond simply identifying locations that a patient may have visited, the physician needs to delve deeper to learn what kinds of activities and behaviors the patient engaged in during travel (e.g., the types of food and sources of water consumed, freshwater swimming, animal exposures) and whether the patient had the necessary immunizations and/or took the necessary prophylactic medications prior to travel; these additional exposures, which the patient may not think to report without specific prompting, are as important as exposures during a patient’s routine daily living.Host-Specific Factors Because many opportunist"  # noqa: E501
    ],
}

b7 = {
    "question.text": "a year old female with a history of type ii diabetes mellitus presents to the emergency department complaining of blood in her urine left sided flank pain nausea and fever she also states that she has pain with urination vital signs include temperature is deg f deg c blood pressure is mmhg pulse is min respirations are and oxygen saturation of on room air on physical examination the patient appears uncomfortable and has tenderness on the left flank and left costovertebral angle which of the following is the next best step in management",  # noqa: E501
    "answer.target": 0,
    "answer.text": ["Obtain a urine analysis and urine culture"],
    "answer.synonyms": [
        "Urinalysis",
        "Acquisition (action)",
        "Obtain",
        "Urine for culture",
    ],
    "document.text": [
        "ncreases systemic blood pressure. Decreased venous return from the placenta decreases right atrial pressure. As breathing begins, air replaces lung fluid, maintaining the functional residual capacity. Fluid leaves the lung, in part, through the trachea; it is either swallowed or squeezed out during vaginal delivery. The pulmonary lymphatic and venous systems reabsorb the remaining fluid.Most normal infants require little pressure to spontaneously open the lungs after birth (5 to 10 cm H2O). With the onset of breathing, pulmonary vascular resistance decreases, partly a result of the mechanics of breathing and partly a result of the elevated arterial oxygen tensions. The increased blood flow to the lungs increases the volume of pulmonary venous blood returning to the left atrium; left atrial pressure now exceeds right atrial pressure, and the foramen ovale closes. As the flow through the pulmonary circulation increases and arterial oxygen tensions rise, the ductus arte"  # noqa: E501
    ],
}

b8 = {
    "question.text": "a year old man with transitional cell carcinoma of the bladder comes to the physician because of a day history of ringing sensation in his ear he received this first course of neoadjuvant chemotherapy week ago pure tone audiometry shows a sensorineural hearing loss of db the expected beneficial effect of the drug that caused this patient s symptoms is most likely due to which of the following actions",  # noqa: E501
    "answer.target": 0,
    "answer.text": ["Ketotifen eye drops"],
    "answer.synonyms": ["DNA Crosslinking"],
    "document.text": [
        "n eukaryotes requires condensation of chromatin.C. in prokaryotes is accomplished by a single DNA polymerase.D. is initiated at random sites in the genome.E. produces a polymer of deoxyribonucleoside monophosphates linked by 5′→3′-phosphodiester bonds. . What is the difference between DNA proofreading and repair?Case 6: Dark Urine and Yellow ScleraePatient Presentation: JF is a 13-year-old boy who presents with fatigue and yellow sclerae.Focused History: JF began treatment ~4 days ago with a sulfonamide antibiotic and a urinary analgesic for a urinary tract infection. He had been told that his urine would change color (become reddish) with the analgesic, but he reports that it has gotten darker (more brownish) over the"  # noqa: E501
    ],
}

b9 = {
    "question.text": "a junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician during the case the resident inadvertently cuts a flexor tendon the tendon is repaired without complication the attending tells the resident that the patient will do fine and there is no need to report this minor complication that will not harm the patient as he does not want to make the patient worry unnecessarily he tells the resident to leave this complication out of the operative report which of the following is the correct next action for the resident to take",  # noqa: E501
    "answer.target": 0,
    "answer.text": ["Tell the attending that he cannot fail to disclose this mistake"],
    "answer.synonyms": [],  # noqa: E501
    "document.text": [
        "professional norms of medicine (the Hippocratic oath, respect to patients and colleagues, ethical conduct, personal accountability, empathy, and altruism) are modeled in every personal encounter. It is imperative that all resident and attending surgeons under-stand that the medical students are observing them closely. When resident and attending surgeons model professional behavior, the hidden curriculum becomes a useful tool for professional devel-opment.147-150 This consistent modeling of professional behavior is one necessary component of leadership.During their clinical years, medical students experience both an exponential growth in knowledge and a measurable decline in empathy towards their patients. Initially, medical stu-dents are filled with excitement and wonder during their first patient encounters. The rapid pace of clinical work, acquisition of knowledge, and intense experiences create stress for the stu-dent, both positively and negatively. Scrubbing into the operat-ing room, witn"  # noqa: E501
    ],
}  # noqa: E501

b10 = {"question.text": "An 82-year-old woman comes to the physician because of difficulty sleeping and increasing fatigue. Over the past 3 months she has been waking up early and having trouble falling asleep at night. During this period, she has had a decreased appetite and a 3.2-kg (7-lb) weight loss. Since the death of her husband one year ago, she has been living with her son and his wife. She is worried and feels guilty because she does not want to impose on them. She has stopped going to meetings at the senior center because she does not enjoy them anymore and also because she feels uncomfortable asking her son to give her a ride, especially since her son has had a great deal of stress lately. She is 155 cm (5 ft 1 in) tall and weighs 51 kg (110 lb); BMI is 21 kg/m2. Vital signs are within normal limits. Physical examination shows no abnormalities. On mental status examination, she is tired and has a flattened affect. Cognition is intact. Which of the following is the most appropriateinitial step in management?",
       "answer.target": 0,
       "answer.text": ["Assess for suicidal ideation"],
       "answer.synonyms": [],
       "document.text": [
           "to the 19th centurywestern medical science\'s understanding and construction of postpartum depression has evolved over the centuries. ideas surrounding women\xe2\x80\x99s moods and states have been around for a long time, typically recorded by men. in 460 b.c., hippocrates wrote about puerperal fever, agitation, delirium, and mania experienced by women after child birth. hippocrates\' ideas stilllinger in how postpartum depression is seen today.a woman who lived in the 14th century, margery kempe, was a christian mystic. she was a pilgrim known as "madwoman"after having a tough labor and delivery. there was a long physical recovery period during which she started descending into "madness" and became suicidal. based on her descriptions of visions of demons and conversations she wrote about that she had withreligious figures like god and the virgin mary, historians have identified what margery"
       ]}

b11 = {"question.text": "A 52-year-old man presents his primary care physician for follow-up. 3 months ago, he was diagnosed with type 2 diabetes mellitus and metformin was started. Today, his HbA1C is 7.9%. The physician decides to add pioglitazone for better control of hyperglycemia. Which of the following is a contraindication to pioglitazone therapy?",
       "answer.target": 0,
       "answer.text": ["History of bladder cancer"],
       "answer.synonyms": [],
       "document.text": [
           "de. his antihyperten-sive therapy was optimized and his urine albumin excretion declined to 1569 mg/g creatinine. this case illustrates the importance of weight loss in controlling glucose levels in the obese patient with type 2 diabetes. it also shows that simply increasing the insulin dose is not always effective. combin-ing metformin with other oral agents and non-insulin inject-ables may be a better option.daniel d. bikle, md, phda 65-year-old man is referred to you from his primary care physician (pcp) for evaluation and management of pos-sible osteoporosis. he saw his pcp for evaluation of low back pain. x-rays of the spine showed some degenerativechanges in the lumbar spine plus several wedge deformities in the thoracic spine. thepati"
       ]}

b12 = {"question.text": "A 70-year-old man is brought to the emergency department with complaints of chest pain for the last 2 hours. He had been discharged from the hospital 10 days ago when he was admitted for acute myocardial infarction. It was successfully treated withpercutaneous coronary intervention. During the physicalexam, the patient prefers to hunch forwards as this decreases his chest pain. He says the pain is in the middle of the chest and radiates to his back. Despite feeling unwell, the patient denies any palpitations or shortness of breath. Vitals signs include: pulse 90/min, respiratory rate 20/min, blood pressure 134/82 mm Hg, and temperature 36.8°C (98.2°F). The patient is visibly distressed and is taking shallow breaths because deeper breaths worsen his chest pain. An ECG shows diffuse ST elevations. Which of the following should be administered to this patient?",
       "answer.target": 0,
       "answer.text": ["Ibuprofen"],
       "answer.synonyms": [],
       "document.text": [
           "welling had diminished and there was a noted improvement in range of motion of the third metacarpophalangeal joint.unnamed 16-year old malea 16-year old teenage male was seen for sudden pain in his right metacarpophalangeal joints. though there was no history of trauma, the patient was a manual laborer.range of motion was slightly limited and joint was mildly swollen and tender when palpated. patient was originally treated with splinting and ibuprofen, but this further worsened his condition. patient was then treated with physical therapy, but symptoms persisted. finally, patient was treated with bone grafting surgeryand splinted for three weeks. after surgery followed byphysical therapy, full range of motion was restored within eight weeks.unnamed 36-year old malea 36-year old male electrician with no past history of trauma presented with a painful right middle finger metacarpophalangeal joint. rang"
       ]}

b13 = {"question.text": "A 7-year-old boy is brought to the pediatrician by his parents for concern of general fatigue and recurrent abdominal pain. You learn that his medical history is otherwise unremarkable and that these symptoms started about 3 months ago after they moved to a different house. Based on clinical suspicionlabs are obtained that reveal a microcytic anemia with high-normal levels of ferritin. Examination of a peripheral blood smear shows findings that are demonstrated in the figure provided. Which of the following is the most likely mechanism responsible for the anemia in this patient?",
       "answer.target": 0,
       "answer.text": ["Inhibition of ALA dehydratase and ferrochelatase"],
       "answer.synonyms": [],
       "document.text": [
           "testinal problems. the parents report that the boy has been listless for the last few weeks. lab tests reveal a microcytic, hypochromic anemia. blood lead levels are elevated. which of the enzymes listed below is most likely to have higher-than-normal activity in the liver of this child?a. \xce\xb4-aminolevulinic acid synthaseb. bilirubin udp glucuronosyltransferasec. ferrochelatased. heme oxygenasee. porphobilinogen synthasecorrect answer = a.this child has the acquired porphyria of lead poisoning. lead inhibits both \xce\xb4-aminolevulinic acid dehydratase and ferrochelatase and, consequently, heme synthesis. the decrease in heme derepresses \xce\xb4aminolevulinic acid synthase-1 (the h"
       ]}

b14 = {"question.text": "A 36-year-old G2-P1 woman in week 33 of gestation presents to the emergency department in acuterespiratory distress. She works as a secretary for a local law firm, and she informs you that she recently returned from a trip to the beach. She currently smokeshalf-a-pack of cigarettes/day, drinks 1 glass of red wine/day, and she endorses a past history of injection drug use but currently denies any illicit drug use. Thevital signs include: temperature 36.7°C (98.0°F), bloodpressure 126/74 mm Hg, heart rate 87/min, and respiratory rate 23/min. Her physical examination showsminimal bibasilar rales, but otherwise clear lungs on auscultation, grade 2/6 holosystolic murmur, and a gravid uterus with no obvious abnormalities. A D-dimer is found to be elevated, and her V/Q scan reveals a high probability of pulmonary embolism (PE). Her medical history is significant for uterine fibroids, preeclampsia, hypercholesterolemia, diabetes mellitus type 1, and significant for heparin-induced thrombocytopenia. Which of the following is the most appropriate choice of management for her post-acute care?",
       "answer.target": 0,
       "answer.text": ["Consult IR for IVC filter placement"],
       "answer.synonyms": [],
       "document.text": [
           "one such family has been reported. clinical features habib et al. (2019) reported a 66-year-old caucasian woman with a lifelong history of insensitivity to pain,even after surgery, including hip replacement and shoulder surgery for osteoarthritis. she did not require analgesia for varicose vein and dental procedures and reported numerous burns and cuts withoutpain, which she stated healed quickly with little or noresidual scarring. she could tolerate eating hot peppers with a short-lasting 'pleasant glow' in her mouth, but no discomfort. she described sweating normally in warm conditions. she also scored very low on tests for anxiety and reported never panicking in dangerous or fearful situations, although she had a long history of short memory lapses, such as forgettingwords or location of keys. physical examination showed multiple scars on the arms and on the back of her hands, and sensory testing demonstrated hyposensitivi"
       ]}

b15 = {"question.text": "ou have been asked to deliver a lecture to medical students about the effects of various body hormones and neurotransmitters on the metabolism of glucose. Which of the following statements best describes the effects of sympathetic stimulation on glucose metabolism?",
       "answer.target": 0,
       "answer.text": ["Epinephrine increases liver glycogenolysis"],
       "answer.synonyms": [],
       "document.text": [
           "oms that are rapidly resolved by the administration of glucose. insulin-induced, postprandial, and fasting hypoglycemia result in release of glucagon and epinephrine. the rise in nicotinamide adenine dinucleotide (nadh) that accompanies ethanolmetabolism inhibits gluconeogenesis, leading to hypoglycemia in individuals with depleted stores. alcohol consumption also increases the risk for hypoglycemia in patients using insulin. chronic alcohol consumption can cause fatty liver disease.choose the one best answer.3.1. which of the following statements is true for insulin but not for glucagon?a. it is a peptide hormone secreted by pancreatic cells.b. its actions are mediated by binding to a receptor found on the cell membrane of liver cells.c. its effects include alterations in gene expression.d. its secretion i"
       ]}

b16 = {"question.text": "A 45-year-old man presents to his primary care physician for a general checkup.The patient has no complaints, but is overweight by 20 lbs. The physician orders outpatient labs which come back with an elevated total bilirubin. Concerned, the PCP orders further labs which show: total bilirubin: 2.4, direct bilirubin 0.6, indirect bilirubin 1.8. Which of the following are true about this patient's condition?",
       "answer.target": 0,
       "answer.text": ["Diagnosis is readily made with characteristic metabolic response to rifampin"],
       "answer.synonyms": [],
       "document.text": [
           "bert's syndrome can be classed as a minor inborn error of metabolism.diagnosispeople with gs predominantly have elevated unconjugated bilirubin, while conjugated bilirubin is usually within the normal range and is less than 20% of the total. levels of bilirubin in gs patients are reported to be from 20 \xce\xbcm to 90 \xce\xbcm (1.2 to 5.3\xc2\xa0mg/dl) compared to the normal amount of &lt; 20 \xce\xbcm. gspatients have a ratio of unconjugated/conjugated (indirect/direct) bilirubin commensurately higher than those without gs.the level of total bilirubin is often further increased if the blood sample is taken after fasting for two days, and a fast can, therefore, be useful diagnostically. a further conceptual step that is rarely necessary orappropriate is to give a low"
       ]}

b17 = {"question.text": "A 3-month-old girl is brought to the emergency department by her parents after she appeared to have a seizure at home. On presentation, she no longer has convulsions though she is still noted to be lethargic. She was born through uncomplicated vaginal delivery and was not noted to have any abnormalities at the time of birth. Since then, shehas been noted by her pediatrician to be falling behind in height and weight compared to similarly aged infants. Physical exam reveals an enlarged liver, and laboratory tests reveal a glucose of 38 mg/dL. Advanced testing shows that a storage molecule present in the cells of this patient has abnormally short outer chains. Which of the following enzymes is most likely defective in this patient?",
       "answer.target": 0,
       "answer.text": ["Debranching enzyme"],
       "answer.synonyms": [],
       "document.text": [
           "atal period with signs of an increased bleeding tendency.  huybrechts et al. (2012) reported a 27-month-old girl, born of consanguineous moroccan parents, with cdg2l. at birth, she was noted to have dysmorphic features, including microcephaly, postaxial polydactyly, broad palpebral fissures, retrognathia, and anal anteposition. during the first months of life, she had recurrent infections, diarrhea, and failure to thrive, and was found to have a primary combined immunodeficiency with hypogammaglobulinemia and defective cellular immunity without lymphopenia. granulocyte function was also abnormal. she later developed multisystem abnormalities, including hepatomegaly, abnormal liver enzymes, micronodular cirrho"
       ]}

b18 = {"question.text": "An 11-year-old boy is brought to the emergency department by his parents with a2-day history of fever, malaise, and productive cough. On presentation, he is found to be very weak and is having difficulty breathing. His past medical history is significant for multiple prior infections requiring hospitalization including otitis media, upper respiratory infections, pneumonia, and sinusitis. His family history is also significant for a maternal uncle who died of an infection as a child. Lab findings include decreased levels of IgG, IgM, IgA, and plasma cells with normal levels of CD4 positive cells. The protein that is most likely defective in this patient has which of the following functions?",
       "answer.target": 0,
       "answer.text": ["Protein phosphorylation"],
       "answer.synonyms": [],
       "document.text": [
           ""
       ]}

b19 = {"question.text": "",
       "answer.target": 0,
       "answer.text": [""],
       "answer.synonyms": [],
       "document.text": [
           ""
       ]}


exs = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9]
batch = Collate()(exs)
pprint_batch(batch, header="Input batch")

with Status("Loading classifiers.."):
    classifiers = [
        ExactMatch(interpretable=True),
        MetaMapMatch(interpretable=True, lazy_setup=False),
        ScispaCyMatch(interpretable=True, lazy_setup=False),
    ]

with Status("Processing examples.."):
    output = {c: c(copy(batch)) for c in classifiers}

for i, eg in enumerate(exs):
    print(get_separator())
    rich.print(f"[cyan]Question: {eg['question.text']}")
    rich.print(f"[red]Answer: {eg['answer.text'][eg['answer.target']]}")
    rich.print(f"[white]Document: {eg['document.text'][0]}")
    for c, b in output.items():
        _match_on = b["document.match_on"][i][0] if "document.match_on" in b else None
        rich.print(
            f"> {type(c).__name__}: match_score={b['document.match_score'][i][0]}, "
            f"match_on={_match_on}"
        )
