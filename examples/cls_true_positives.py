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
    "answer.text": ["Post polio syndrome"],
    "answer.cui": [],
    "document.text": [
        """Post polio syndrome is a condition that affects polio survivors years after recovery from the initial
        polio illness. Symptoms and severity vary among affected people and may include muscle weakness and a
        gradual decrease in the size of muscles (atrophy); muscle and joint pain; fatigue;difficulty with gait;
        respiratory problems; and/or swallowing problems.\xa0Only a polio survivor can develop PPS. While polio
        is a contagious disease, PPS is not. The exact cause of PPS years after the first episode of polio is
        unclear, although several theories have been proposed. Treatment focuses on reducing symptoms and improving
        quality of life."""  # noqa: E501
    ],
}

b1 = {
    "question.text": """A 59-year-old overweight woman presents to the urgent care clinic with the complaint of
        severe abdominal pain for the past 2 hours. She also complains of a dull pain in her back
        with nausea and vomiting several times. Her pain has no relation with food. Her past medical
        history is significant for recurrent abdominal pain due to cholelithiasis. Her father died at
        the age of 60 with some form of abdominal cancer. Her temperature is 37\u00b0C (98.6\u00b0F),
        respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. Physical exam is
        unremarkable. However, a CT scan of the abdomen shows a calcified mass near her gallbladder.
        Which of the following diagnoses should be excluded first in this patient?""",  # noqa: E501
    "answer.target": 1,
    "answer.text": [
        "Acute cholecystitis",
        "Gallbladder cancer",
        "Choledocholithiasis",
        "Pancreatitis",
    ],
    "answer.cui": ["C0235782"],
    "document.text": [
        """Carcinoma of the gallbladder (GBC) is the most common and aggressive form of biliary tract
        cancer (BTC; see this term) usually arising in the fundus of the gallbladder, rapidly metastasizing
        to lymph nodes and distant sites. Epidemiology Annual incidence rates vary from 1/100,000 to 1/ 4,350
        between different ethnic groups and geographical regions. It is rare in developed Western countries
        but has a high incidence in Japan (1/19,000), northern India, Chile and certain regions of Eastern
        Europe. Clinical description GBC is a rare neoplasm occurring more often in females (3-4:1 female
        to male ratio) with an average age of onset of 65 years. Most patients are asymptomatic until the
        disease is advanced but presenting symptoms include abdominal pain (usually in the upper right
        quadrant), nausea, vomiting, jaundice, anorexia and weight loss. Gallstones are often present in
        patients with GBC. GBC is extremely aggressive and invasion of the lymph nodes, liver and other
        organs occurs rapidly in many cases. Etiology The exact etiology is unknown. Genetic susceptibility
        elicited by chronic inflammation of the gallbladder leading to dysplasia and malignant change is one
        possibility. Risk factors associated with GBC include a history of gallstones, cholelithiasis, porcelain
        gallbladder, bacterial infections, high caloric diet"""  # noqa: E501
    ],
}

b2 = {
    "question.text": """A 67-year-old man who was diagnosed with arthritis 16 years ago presents with right knee
        swelling and pain. His left knee was swollen a few weeks ago, but now with both joints affected,
        he has difficulty walking and feels frustrated. He also has back pain which makes it extremely
        difficult to move around and be active during the day. He says his pain significantly improves
        with rest. He also suffers from dandruff for which he uses special shampoos. Physical examination
        is notable for pitting of his nails. Which of the following is the most likely diagnosis?""",  # noqa: E501
    "answer.target": 0,
    "answer.text": [
        "Psoriatic arthritis",
        "Arthritis mutilans",
        "Rheumatoid arthritis",
        "Mixed connective tissue disease",
    ],
    "answer.cui": ["C0003872"],
    "document.text": [
        """the fingers, nails, and skin. Sausage-like swelling in the fingers or toes, known as dactylitis,
        may occur. Psoriasis can also cause changes to the nails, such as pitting or separation from the
        nail bed, onycholysis, hyperkeratosis under the nails, and horizontal ridging. Psoriasis classically
        presents with scaly skin lesions, which are most commonly seen over extensor surfaces such as the scalp,
         natal cleft and umbilicus. In psoriatic arthritis, pain can occur in the area of the sacrum (the lower
         back, above the tailbone), as a result of sacroiliitis or spondylitis, which is present in 40% of cases.
         Pain can occur in and around the feet and ankles, especially enthesitis in the Achilles tendon
         (inflammation of the Achilles tendon where it inserts into the bone) or plantar fasciitis in the sole of
         the foot. Along with the above-noted pain and inflammation, there is extreme exhaustion that does not
         go away with adequate rest. The exhaustion may last for days or weeks without abatement. Psoriatic
         arthritis may remain mild or may progress to more destructive joint disease. Periods of active disease,
         or flares, will typically alternate with periods of remission. In severe forms, psoriatic arthritis may
         progress to arthritis mutilans which on"""  # noqa: E501
    ],
}

b3 = {
    "question.text": """A 29-year-old man is admitted to the emergency department following a motorcycle accident.
        The patient is severely injured and requires life support after splenectomy and evacuation of
        a subdural hematoma. Past medical history is unremarkable. The patient’s family members, including
        wife, parents, siblings, and grandparents, are informed about the patient’s condition. The patient has no
        living will and there is no durable power of attorney. The patient must be put in an inducedcoma for an
        undetermined period of time. Which of the following is responsible for making medical decisions for the
        incapacitated patient?""",
    "answer.target": 0,
    "answer.text": ["The spouse"],
    "answer.cui": [],
    "document.text": [
        """ate) orders.durable power of attorney for health care (dpoahc): legally designates a surrogate health
        care decision maker if a patient lacks decision-making capacity. more \xef\xac\x82 exible than a living
        will. surrogates should make decisions consistent with the person\xe2\x80\x99s stated wishes.if no living
        will or dpoahc exists, decisions should be made by close family members (spouse, adult children, parents,
        and adult siblings), friends, or personal physicians, in that order.in the absence of a living will or
        dpoa, the spouse chips in for the person: spouse, children, parent, sibling, friend, personal physician.
        dnr/dni orders do not mean \xe2\x80\x9cdo not treat.\xe2\x80\x9dpatients cannot demand futile treatment
        from their physicians.a patient\xe2\x80\x99s family cannot require that a doctor withhold information from
        the patient.withdrawal of carepatients and their decision makers have the right to forgo or w"""
    ],
}

b4 = {
    "question.text": """A 28-year-old man comes to the physician for a follow-up examination after a previous
    visit showed an elevated serum calcium level. He has a history of bipolar disorder. His mother had a
    parathyroidectomy inher 30s. The patient does not drink alcohol or smoke. Current medications include
    lithium and a daily multivitamin. His vital signs are within normal limits.Physical examination shows no
    abnormalities. Laboratorystudies show:SerumSodium 146 mEq/LPotassium 3.7 mEq/LCalcium 11.2 mg/dLMagnesium
    2.3 mEq/LPTH 610 pg/mLAlbumin 4.2 g/dL24-hour urineCalcium 23 mgWhich of the following is the most likely
    cause of thispatient’s findings?""",
    "answer.target": 0,
    "answer.text": ["Abnormal calcium sensing receptors"],
    "answer.cui": [],
    "document.text": [
        """(1.1 to  1.3 mmol/l). both concentrations are tightly regulated. the total serum calcium level must
        always be considered in its rela-tionship to plasma protein levels, especially serum albumin. for each
        gram per deciliter of alteration of serum albumin above or below 4.0 mg/dl, there is a 0.8 mg/dl increaseor
        decrease in protein-bound calcium and, thus, in total serum calcium levels. total and, particularly,
        ionized calcium levels are influenced by various hormone systems.parathyroid hormone.\xe2\x80\x82the
        parathyroid cells rely on a h-protein\xe2\x80\x93coupled membrane receptor, designated the calcium-sensing
        receptor (casr), to regulate pth secretion by sensing extracellular calciumlevels69 (fig. 38-28). pth
        secretion also is stimulatedby low levels of 1,25-dihydroxy vit"""
    ],
}

b5 = {
    "question.text": """Twenty minutes after delivery of a newborn infant, a 22-year-old woman starts breastfeeding.
    Initially, the expressed milk is thick and yellowish. Three days later, themother's breasts swell and the
    expressed milk becomes thinner and whiter. A decrease in maternal serum concentration of which of the following
    is most likely responsible for the observed changes in milk production?""",
    "answer.target": 0,
    "answer.text": ["Progesterone"],
    "answer.cui": [],
    "document.text": [
        """by 24 mo)defined as < 2500 g. caused by prematurity or intrauterine growth restriction (iugr). associated
        with \xc2\x8f\xc2\xa0risk of sudden infant death syndrome (sids) and with \xc2\x8f overall mortality.lactation
        after parturition and delivery of placenta, rapid \xc2\x90 in progesterone disinhibits prolactin \xe2\x80\xa2
        initiation of lactation. suckling is required to maintain milk production and ejection, since \xc2\x8f nerve
        stimulation \xc2\x8e\xc2\x8f\xc2\xa0oxytocin and prolactin. prolactin\xe2\x80\x94induces and maintains
        lactation and \xc2\x90 reproductive function. oxytocin\xe2\x80\x94assists in milk letdown; also promotes
        uterine contractions. breast milk is the ideal nutrition for infants < 6 months old. contains maternal
        immunoglobulins (conferring passive immunity; mostly iga), macrophages, lymphocytes. breast milk reduces
        infant infections and is associated with \xc2\x90 risk for chil"""
    ],
}

b6 = {
    "question.text": """A 38-year-old man arrives at the emergency department with severe periumbilical, colicky pain and abdominal
        distention for the past 2 days. He is nauseated and reports vomiting light-green emesis 8 times since yesterday
        morning. He has not had a bowel movement or passed any gas for the past 3 days. He has a past medical history
        of ventral hernia repair 5 years ago. His heart rate is 110/min, respiratory rate is 24/min, temperature is
        38.0°C (100.4°F), and blood pressure is 120/90 mm Hg. The abdomen is distended and mildly tender to deep
        palpation. Bowel sounds are high-pitched and tinkling.Which of the following is the most likely cause?""",
    "answer.target": 0,
    "answer.text": ["Small bowel obstruction"],
    "answer.cui": [],
    "document.text": [
        """ope in someone with a small bowel obstruction.problems playing this file? see media help.depending on
        the level of obstruction, bowel obstruction can present with abdominal pain, swollen abdomen, abdominal
        distension, and constipation. bowel obstructionmay be complicated by dehydration and electrolyte abnormalities
        due to vomiting; respiratory compromise from pressure on the diaphragm by a distended abdomen, or aspiration
        of vomitus; bowel ischemia or perforation from prolonged distension or pressurefrom a foreign body.in small
        bowel obstruction, the pain tends to be colicky (cramping andintermittent) in nature, with spasms lasting a
        few minutes. the pain tends to be central and mid-abdominal. vomiting may occur before constipation.in large
        bowel obstruction, the pain is felt lower in the abdo"""
    ],
}

b7 = {
    "question.text": """A 27-year-old man comes to the physician with throbbing right scrotal pain for 1 day. He has also had a
        burning sensation on urination during the last 4 days. He is sexually active with multiple female partners
        and does not use condoms. Physical examination shows a tender, palpable swelling on the upper pole of the
        right testicle; lifting the testicle relieves the pain. A Gram stain of urethral secretions shows
        numerouspolymorphonuclear leukocytes but no organisms. Which of the following is the most likely causal
        pathogen of this patient's symptoms?""",
    "answer.target": 0,
    "answer.text": ["Chlamydia trachomatis"],
    "answer.cui": [],
    "document.text": [
        """r urethrapreventionnot having sex, condoms, sex with only one non\xe2\x80\x93infected persontreatment
        antibiotics (azithromycin or doxycycline)frequency4.2% (women), 2.7% (men)deaths~200 (2015)chlamydia, or
        more specifically a chlamydia infection, is a sexually transmitted infection caused by the bacterium
        chlamydia trachomatis. most people who are infected have no symptoms. when symptoms do appear it can be
        several weeks after infection. symptoms in women may include vaginal discharge or burning with urination.
        symptoms in men may include discharge from the penis, burning with urination, or pain and swelling of one
        or both testicles. the infection can spread to the upper genital tract in women, causing pelvic inflammatory
        disease, which may result in future infertility or ectopic pregnancy. rep"""
    ],
}

b8 = {
    "question.text": """A 45-year-old man with a history of recurrent gouty arthritis comes to the physician for a follow-up
        examination. Four weeks ago, he was diagnosed with hyperuricemiaand treatment with allopurinol was begun.
        Since then, he has had another acute gout attack, which resolved after treatment with ibuprofen. His
        temperature is 37.1°C (98.8°F).Physical examination shows painless, chalky nodules on the metatarsophalangeal
        joint of his right foot. Laboratory studies show: Serum Creatinine 1.0 mg/dL Uric acid 11.6 mg/dL Cholesterol
        278 mg/dL Urine Uric acid 245 mg/24 h (N = 240-755)Based on the urine findings, this patient would most likely
        benefit from treatment with which of the following drugs to prevent future gout attacks?""",
    "answer.target": 0,
    "answer.text": ["Probenecid"],
    "answer.cui": [],
    "document.text": [
        """two attacks), serum uric acid levels (progression is more rapid in patients with serum uric acid >535
        \xce\xbcmol/l [>9.0 mg/dl]), the patient\xe2\x80\x99s willingness to commit to lifelong therapy, or the
        presence of uric acid stones. urate-lowering therapy should be initiated in any patient who already has
        tophi or chronicgouty arthritis. uricosuric agents such as probenecid can be used in patients with good
        renal function who underexcrete uric acid, with <600 mg in a 24-h urine sample. urine volume should be
        maintained by ingestion of 1500 ml of water every day. probenecid can be started at a dose of 250 mg twice
        daily and increased gradually as needed up to 3 g per day to achieve and maintain a serum uric acid level
        of less than 6 mg/dl. probenecid is generally not effe"""
    ],
}

b9 = {
    "question.text": """A 33-year-old woman is brought to the physician by her husband because of persistent sadness for the
        past 2 months. During this period, she also has had difficulty sleeping and an increased appetite. She had
        similar episodes that occurred 2 years ago and9 months ago that each lasted for 4 months. Between these
        episodes, she reported feeling very energetic and rested after 3 hours of sleep. She often went for long
        periods of time without eating. She works as a stock market trader and received a promotion 5 months ago.
        She regularly attends yoga classes on the weekends with her friends. On mental status examination, she has
        a blunted affect. She denies suicidal thoughts and illicit drug use. Which of the following is the most
        likely diagnosis?""",
    "answer.target": 0,
    "answer.text": ["Cyclothymic disorder"],
    "answer.cui": [],
    "document.text": [
        """aracterized by periods of de- pressive symptoms and periods of hypomanic symptoms, the lifetime presence
        of a major depressive episode precludes the diagnosis of cyclothymic disorder.short-duration hypomania,
        similar to full hypomanic episodes, has been associated with higher rates of comorbid anxiety disorders and
        substance use disorders than are found in the general population.a. the individual experienced the deathof
        someone with whom he or she had a close re- lationship.b. since the death, at least oneof the following
        symptoms is experienced on more days than not and to a clinically significant degree and has persisted for
        at least 12 months after the death in the case ofbereaved adults and 6 months for bereaved children: 1.
        persistent yearning/ionging for thedeceased. in young children, yearning may be expressed in play and behavior,
        including behaviors that reflect being separated from, and"""
    ],
}

b10 = {
    "question.text": """The incidence of a relatively benign autosomal recessive disease, X, is 1 in 25 in the population.
        Assuming that the conditions for Hardy Weinberg Equilibrium are met, what is the probability that a
        male and female, who are carriers, will have a child expressing the disease?""",
    "answer.target": 0,
    "answer.text": ["1/4"],
    "answer.cui": [],
    "document.text": [
        """4, p. 289). if a couple has a child with an autosomal recessive disease, the recurrence risk is 25
        percent for each subsequent pregnancy. hus, 1/4 of ofspring will be homozygous normal, 2/4 will be
        heterozygous carriers, and 114 will be homozygous abnormal.in other words, three of four children will
        be phenotypically normal, and 2/3 of phenotypically normal siblings are actually carriers.a heterozygous
        carrier of a recessivecondition is only at risk to have afected children if his or her partner is
        heterozygous or homozygous for the disease. genes for rare autosomal recessive conditions have low
        prevalence in the general population. thus, the likelihood that a partner will be a gene carrier is small,
        unless there is consanguinity or the part"""
    ],
}

b11 = {
    "question.text": """A 68-year-old man comes to the physician because of a 4-month history of difficulty swallowing. During
        this time, he has also had a 7-kg (15-lb) weight loss. Esophagogastroduodenoscopy shows an exophytic mass
        in the distal third of the esophagus. Histological examination of a biopsy specimen shows a well-differentiated
        adenocarcinoma. The patient is scheduled for surgical resection of the tumor. During the procedure, the surgeon
         damages a structure that passes through the diaphragm along with the esophagus at the level of the tenth
         thoracic vertebra (T10). Which of the following structures was mostlikely damaged?""",
    "answer.target": 0,
    "answer.text": ["Vagus nerve"],
    "answer.cui": [],
    "document.text": [
        """x and abdomen pass through the diaphragm or between the diaphragm and its peripheral attachments:the
        inferior vena cava passes through the central tendon at approximately vertebral level tviii.the esophagus
        passes through the muscular part of the diaphragm, just to the left of midline, approximately at vertebral
        level tx.the vagus nerves pass through the diaphragm with the esophagus.the aorta passes behind the
        posteriorattachment of the diaphragm at vertebral level txii.the thoracic duct passes behind the diaphragm
        with the aorta.the azygos and hemiazygos veins may also pass through the aortic hiatus or through the crura
        of the diaphragm.other structures outside the posterior attachments of the diaphragm lateral to the aortic
        hiatus incl"""
    ],
}

b12 = {
    "question.text": """A 42-year-old man presents to the emergency department with a 3-day history of fever and severe back pain.
        The fever is high-grade, continuous, without chills and rigors. The back pain is severe, localized to the
        thoracic region, and aggravated by deep breathing. The patient tried taking ibuprofen with little improvement.
        Past medical history is significant for essential hypertension, dyslipidemia, hyperuricemia, and bronchial
        asthma. Current medicines include allopurinol, amlodipine, atorvastatin, clopidogrel, montelukast, and a
        corticosteroid inhaler. The patient reports a 25-pack-year smoking history and drinks alcohol only socially.
        His vital signs include: bloodpressure 152/94 mm Hg, pulse 101/min, temperature 39.5\xc2\xb0C (103.1\xc2\xb0F).
        BMI 36.8 kg/m2. On physical examination, the patient is alert and oriented. Multiple injection marks are
        visible around the left-sided cubitalfossa and hand veins. The neck is supple on head flexion. Point
        tenderness is present in the thoracic region at the midline. Motor and sensory examinations are unremarkable
        with normal deep tendon reflexes. Laboratory findings are significant for the following:\nHemoglobin 14.5
        mg/dL\nWhite blood cell 24,500/mm3\nPlatelets 480,000/mm3\nBUN 28 mg/dL\nCreatinine 1.1 mg/dL\nESR 45
        mm/hr\nC-reactive protein 84 mg/dL\nSodium 144 mEq/L\nPotassium 4.1 mEq/L\nCalcium 9.7 mEq/L\nA contrast
        MRI of the spine reveals a peripherally enhancing dorsal epidural process compressing the thecal sac and
        causing a mild leftwards displacement. Which of the following is the most likely risk factor for this
        patient\xe2\x80\x99s condition?""",
    "answer.target": 0,
    "answer.text": ["Intravenous drug use"],
    "answer.cui": ["C0699778"],
    "document.text": [
        """118.vertebral osteomyelitis is often caused by staphylococci, but other bacteria or tuberculosis
        (pott\xe2\x80\x99s disease) may be responsible. the primary source of infection is usually the urinary
        tract, skin, or lungs. drug intravenous use is a well-recognized risk factor. whenever pyogenic osteomyelitis
        is found, the possibility of bacterial endocarditis should be considered. back pain unrelieved by rest, spine
        tenderness over the involved spine segment, and an elevated esr are the most common findings in vertebral
        osteomyelitis. fever or an elevated white blood cell count is found in a minority of patients. mri and ct
        are sensitive and specific for early detection of osteomyelitis; ct may be more readily available in
        emergency settings and better tolerated by some patients with severe back pain. the intervertebral d"""
    ],
}

b13 = {
    "question.text": """A 62-year-old man presents with epigastric pain over the last 6 months. He says the pain gets worse with
        food, especially coffee. He also complains of excessive belching. He says he has tried omeprazole recently,
        but it has not helped. No significant past medical history or current medications. On physical examination,
        there is epigastric tenderness present on deep palpation. An upper endoscopy is performed which reveals
        gastric mucosa with signs of mild inflammation and a small hemorrhagic ulcer in the antrum. A gastric biopsy
        shows active inflammation, and the specimen stains positive with Warthin\xe2\x80\x93Starry stain, revealing
        Helicobacter pylori. Which of the following is the next, best step in the management of this patient
        \xe2\x80\x99s condition?""",
    "answer.target": 0,
    "answer.text": ["Give amoxicillin, clarithromycin, and omeprazole"],
    "answer.cui": ["C0002645", "C4077730", "C0028978", "C0055856", "C1947971"],
    "document.text": [
        """curs in the antrum and may be caused by nsaid use or h. pylori infection. often asymptomatic, but associated
        with \xe2\x86\x91 risk of pud and gastric cancer.patients may be asymptomatic or may complain of epigastric
        pain, nausea, vomiting, hematemesis, or melena.upper endoscopy can visualize the gastric lining.h. pylori
        infection can be detected by urease breath test, serum igg antibodies (which indicate exposure, not current
        infection), h. pylori stoolantigen, or endoscopic biopsy.\xe2\x96\xa0\xe2\x86\x93 intake of offending agents.
        antacids, sucralfate, h2 blockers, and/or ppis may help.triple therapy (amoxicillin, clarithromycin,
        omeprazole) to treat h. pylori infection.give prophylactic h2 b"""
    ],
}

b14 = {
    "question.text": """A 77-year-old man presents to his primary care physician with lightheadedness and a feeling that he is
        going to pass out. He has a history of hypertension that is treated with captopril. In the office, his
        temperature is 38.3\xc2\xb0C (100.9\xc2\xb0F), the pulse is 65/min, and the respiratory rateis 19/min. His
        sitting blood pressure is 133/91 mm Hg. Additionally, his supine blood pressure is 134/92 mm Hg and standing
        blood pressure is 127/88 mm Hg. These are similar to his baseline blood pressure measured during previous
        visits. An ECG rhythm strip is obtained in the office. Of the following, what isthe likely cause of his
        presyncope?""",
    "answer.target": 0,
    "answer.text": ["Right bundle branch block"],
    "answer.cui": ["C0344421", "C0085615"],
    "document.text": [
        """to identify the cause; and to assess risk for future episodes and serious harm (table 27-1). the initial
        evaluation should include a detailed history, thorough questioning of eyewitnesses, and a complete physical
        and neurologic examination. blood pressure and heart rate should be measured in the supine position and after
        3 min of standing to determine whether orthostatic hypotension is present. an ecg should be performed if
        there is suspicion of syncope due to an arrhythmia or underlying cardiac disease. relevant electrocardiographic
         abnormalities include bradyarrhythmias or tachyarrhythmias,av block, ischemia, old myocardial infarction,
         long qt syndrome, and right bundle branch block. this initial assessment will lead to the identification
         of a causeof syncope in approximately 50% of patients and also allows stratification of pati""",
    ],
}


exs = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14]

batch = Collate()(exs)
pprint_batch(batch, header="Input batch")

with Status("Loading classifiers.."):
    classifiers = [
        ExactMatch(interpretable=True),
        MetaMapMatch(interpretable=True, filter_tui=False, lazy_setup=False),
        ScispaCyMatch(interpretable=True, filter_tui=False, lazy_setup=False),
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
