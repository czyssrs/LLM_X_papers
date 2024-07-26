# LLM+X: A Survey on LLMs for Finance, Healthcare, and Law
A collection of research papers on Large Language Models for Finance, Healthcare, and Law.

Paper: [https://arxiv.org/abs/2405.01769](https://arxiv.org/abs/2405.01769)

* [Finance](#Finance)
* [Healthcare](#Healthcare)
* [Law](#Law)
* [Ethics](#Ethics)

## Finance

### Tasks and Datasets in Financial NLP
<h4 id="sentiment-analysis">Sentiment Analysis (SA)</h4>

- **(Financial Phrase Bank)** _Good debt or bad debt: Detecting semantic orientations in economic texts_ ```JASIST 2014```
[[Paper](https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.23062)]

- **(FiQA Task 1)** _Aspect-based financial sentiment analysis_ ```Open Challenge - WWW 2018```
[[Homepage](https://sites.google.com/view/fiqa/home)]

- **(TweetFinSent)** _TweetFinSent: A Dataset of Stock Sentiments on Twitter_ ```FinNLP 2022```
[[Paper](https://aclanthology.org/2022.finnlp-1.5/)][[Github](https://github.com/jpmcair/tweetfinsent)]

- **(FinSent)** _Is ChatGPT a Financial Expert? Evaluating Language Models on Financial Natural Language Processing_ ```EMNLP 2023```
[[Paper](https://aclanthology.org/2023.findings-emnlp.58/)]

- **(BloombergGPT sentiment tasks)** _BloombergGPT: A Large Language Model for Finance_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2303.17564)]


<h4 id="information-extraction">Information Extraction (IE)</h4>

<h5 id="named_entity_recognition">Named Entity Recognition (NER)</h5>

- **(FIN3)** _Domain Adaption of Named Entity Recognition to Support Credit Risk Assessment_ ```Proceedings of the Australasian Language Technology Association Workshop 2015```
[[Paper](https://aclanthology.org/U15-1010/)]

- **(BloombergGPT NER tasks)** _BloombergGPT: A Large Language Model for Finance_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2303.17564)]

<h5 id="relation_extraction">Relation Extraction (RE)</h5>

- **(REFinD)** _REFinD: Relation Extraction Financial Dataset_ ```SIGIR 2023```
[[Paper](https://dl.acm.org/doi/10.1145/3539618.3591911)]

<h5 id="event_detection">Event Detection</h5>

- **(EDT)** _Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading_ ```IJCNLP 2021```
[[Paper](https://aclanthology.org/2021.findings-acl.186/)]

<h4 id="question-answering">Question Answering (QA)</h4>

- **(FiQA Task 2)** _Opinion-based QA over financial data_ ```Open Challenge - WWW 2018```
[[Homepage](https://sites.google.com/view/fiqa/home)]

- **(FinQA)** _FinQA: A Dataset of Numerical Reasoning over Financial Data_ ```EMNLP 2021```
[[Paper](https://aclanthology.org/2021.emnlp-main.300/)][[Github](https://github.com/czyssrs/FinQA)]

- **(TAT-QA)** _TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance_ ```ACL 2021```
[[Paper](https://aclanthology.org/2021.acl-long.254/)][[Github](https://github.com/NExTplusplus/TAT-QA)]

- **(DocFinQA)** _DocFinQA: A Long-Context Financial Reasoning Dataset_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2401.06915)]

- **(ConvFinQA)** _ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering_ ```EMNLP 2022```
[[Paper](https://aclanthology.org/2022.emnlp-main.421/)][[Github](https://github.com/czyssrs/ConvFinQA)]

- **(PACIFIC)** _PACIFIC: Towards Proactive Conversational Question Answering over Tabular and Textual Data in Finance_ ```EMNLP 2022```
[[Paper](https://aclanthology.org/2022.emnlp-main.469/)][[Github](https://github.com/dengyang17/PACIFIC)]

- **(FinanceBench)** _FinanceBench: A New Benchmark for Financial Question Answering_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2311.11944)][[Github](https://github.com/patronus-ai/financebench)]

- **(Tab-CQA)** _Tab-CQA: A Tabular Conversational Question Answering Dataset on Financial Reports_ ```ACL 2023```
[[Paper](https://aclanthology.org/2023.acl-industry.20/)][[Github](https://github.com/tjunlp-lab/Tab-CQA)]


<h4 id="stock-movement-prediction">Text-Enhanced Stock Movement Prediction (SMP)</h4>

- **(ACL-14)** _Using Structured Events to Predict Stock Price Movement: An Empirical Investigation_ ```EMNLP 2014```
[[Paper](https://aclanthology.org/D14-1148/)]

- **(ACl-18)** _Stock Movement Prediction from Tweets and Historical Prices_ ```ACL 2018```
[[Paper](https://aclanthology.org/P18-1183/)][[Github](https://github.com/yumoxu/stocknet-code)]

- **(CIKM-18)** _Hybrid Deep Sequential Modeling for Social Text-Driven Stock Prediction_ ```CIKM 2018```
[[Paper](https://dl.acm.org/doi/10.1145/3269206.3269290)][[Github](https://github.com/wuhuizhe/CHRNN)]

- **(BigData-22)** _Accurate stock movement prediction with self-supervised learning from sparse noisy tweets_ ```Big Data 2022```
[[Paper](https://ieeexplore.ieee.org/document/10020720)] [[Github](https://github.com/deeptrade-public/slot)]

- **(Astock)** _Astock: A New Dataset and Automated Stock Trading based on Stock-specific News Analyzing Model_ ```Arxiv 2022```
[[Paper](https://arxiv.org/abs/2206.06606)][[Github](https://github.com/JinanZou/Astock)]

- **(EDT)** _Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading_ ```IJCNLP 2021```
[[Paper](https://aclanthology.org/2021.findings-acl.186/)]
 
<h4 id="other-financial-nlp-tasks">Other Financial NLP Tasks</h4>

<h5 id="other_classification_tasks">Other Classification tasks</h5>

<h6 id="new_headline_classification">News headline classification</h6>

- _Impact of News on the Commodity Market: Dataset and Results_ ```Arxiv 2020```
[[Paper](https://arxiv.org/abs/2009.04202)]

<h6 id="hawkish_dovish_classification">hawkish-dovish classification</h6>

- _Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis_ ```ACL 2023```
[[Paper](https://aclanthology.org/2023.acl-long.368/)][[Github](https://github.com/gtfintechlab/fomc-hawkish-dovish)]

<h6 id="multiple_classification">Multiple classification tasks</h6>

- _Is ChatGPT a Financial Expert? Evaluating Language Models on Financial Natural Language Processing_ ```EMNLP 2023```
[[Paper](https://aclanthology.org/2023.findings-emnlp.58/)]

<h5 id="sentence_boundary_detection">Sentence boundary detection</h5>

- _Sentence Boundary Detection in PDF Noisy Text in the Financial Domain_ ```FinNLP 2019```
[[Homepage](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp/shared-task-finsbd)]


<h5 id="ner_ned">NER+NED</h5>

- **(BloombergGPT NER+NED tasks)** _BloombergGPT: A Large Language Model for Finance_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2303.17564)]

<h5 id="reasoning">Reasoning</h5>

- **(BizBench)** _BizBench: A Quantitative Reasoning Benchmark for Business and Finance_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2311.06602)]

<h5 id="investment-opinion-generation">Investment opinion generation</h5>

- _Beyond Classification: Financial Reasoning in State-of-the-Art Language Models_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2305.01505)][[Github](https://github.com/guijinSON/FIOG)]

<h5 id="summarization">Summarization</h5>

- **(ECTSum)** _ECTSum: A New Benchmark Dataset For Bullet Point Summarization of Long Earnings Call Transcripts_ ```EMNLP 2022```
[[Paper](https://arxiv.org/abs/2311.06602)][[Github](https://github.com/rajdeep345/ECTSum)]



<h4 id="financial-nlp-tasks-under-explored-for-llms">Financial NLP Tasks Under-Explored for LLMs</h4>

<h5 id="financial-fraud-detection">Financial fraud detection</h5>

- _Intelligent financial fraud detection: A comprehensive review_ ```Computers & Security 2016```
[[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167404815001261)]

<h5 id="risk_assessment_and_management">Risk assessment and management</h5>

- _Machine learning and AI for risk management_ ```Disrupting Finance, 2018```
[[Paper](https://link.springer.com/chapter/10.1007/978-3-030-02330-0_3)]

- _A survey on deep learning for financial risk prediction_ ```Quantitative Finance and Economics 2021```
[[Paper](https://www.aimspress.com/article/doi/10.3934/QFE.2021032?viewType=HTML)]

- _Machine learning for financial risk management: a survey_ ```IEEE access 2020```
[[Paper](https://ieeexplore.ieee.org/document/9249416)]

<h5 id="robo_advisor">Robo advisor</h5>

- _Artificial Intelligence for Conversational Robo-Advisor_ ```ASONAM 2018```
[[Paper](https://ieeexplore.ieee.org/document/8508269)]

- _Artificial intelligence in financial services: a qualitative research to discover robo-advisory services_ ```Qualitative Research in Financial Markets 2021```
[[Paper](https://www.emerald.com/insight/content/doi/10.1108/QRFM-10-2020-0199/full/html?skipTracking=true)]

- _Research on Generative Artificial Intelligence for Virtual Financial Robo-Advisor_ ```Academic Journal of Science and Technology 2024```
[[Paper](https://drpress.org/ojs/index.php/ajst/article/view/19151)]

<h5 id="compliance_and_regulations">Compliance and regulations</h5>

- _The Application of Artificial Intelligence in Financial Compliance Management_ ```AIAM 2019```
[[Paper](https://dl.acm.org/doi/abs/10.1145/3358331.3358339)]

- _Classifying sentential modality in legal language: a use case in financial regulations, acts and directives_ ```ICAIL 2017```
[[Paper](https://dl.acm.org/doi/10.1145/3086512.3086528)]

<h5 id="chatbot_services">Chatbot services</h5>

- _Ai-based chatbot service for financial industry_ ```Fujitsu Scientific and Technical Journal 2018```
[[Paper](https://www.fujitsu.com/global/documents/about/resources/publications/fstj/archives/vol54-2/paper01.pdf)]

- _Toward a chatbot for financial sustainability_ ```Sustainability 2021```
[[Paper](https://www.mdpi.com/2071-1050/13/6/3173)]

- _Text-based chatbot in financial sector: A systematic literature review._ ```Data Science in Finance and Economics 2022```
[[Paper](https://www.aimspress.com/article/doi/10.3934/DSFE.2022011?viewType=HTML)]



### Financial LLMs

<h4 id="pretraining_and_downstream_task_finetuning_plms">Pre-Training and Downstream Task Fine-Tuning PLMs.</h4>

- **(FinBERT-19)** _FinBERT: Financial Sentiment Analysis with Pre-trained Language Models_ ```Arxiv 2019```
[[Paper](https://arxiv.org/abs/1908.10063)][[Github](https://github.com/ProsusAI/finBERT)][[Model](https://huggingface.co/ProsusAI/finbert)]

- **(FinBERT-20)** _Finbert: A pretrained language model for financial communications_ ```Arxiv 2020```
[[Paper](https://arxiv.org/abs/2006.08097)][[Github](https://github.com/yya518/FinBERT)][[Model](https://huggingface.co/yiyanghkust/finbert-pretrain)]

- **(FinBERT-21)** _Finbert: A pre-trained financial language representation model for financial text mining_ ```IJCAI 2020```
[[Paper](https://www.ijcai.org/proceedings/2020/622)]

- **(Mengzi-BERTbase-fin)** _Mengzi: Towards lightweight yet ingenious pre-trained models for chinese_ ```Arxiv 2021```
[[Paper](https://arxiv.org/abs/2110.06696)][[Github](https://github.com/Langboat/Mengzi/blob/main/README_en.md)][[Model](https://huggingface.co/Langboat/mengzi-bert-base-fin)]

- **(FLANG)** _When FLUE Meets FLANG: Benchmarks and Large Pretrained Language Model for Financial Domain_ ```EMNLP 2022```
[[Paper](https://aclanthology.org/2022.emnlp-main.148/)][[Github](https://github.com/SALT-NLP/FLANG)][[Model](https://huggingface.co/SALT-NLP/FLANG-BERT)]

- **(BBT-Fin)** _BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2302.09432)][[Github](https://github.com/ssymmetry/BBT-FinCUGE-Applications)][[Model](https://huggingface.co/SuSymmertry/BBT)]

<h4 id="pretraining_LLMs">Pre-training LLMs</h4>

- **(BloombergGPT)** _BloombergGPT: A Large Language Model for Finance_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2303.17564)]

<h4 id="instruction_finetuning_llms">Instruction Fine-Tuning LLMs.</h4>

- **(FinMA)** _PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2306.05443)][[Github](https://github.com/The-FinAI/PIXIU)][[Model](https://huggingface.co/ChanceFocus/finma-7b-nlp)]

- **(Instruct-FinGPT)** _Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models_ ```IJCAI 2023```
[[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4489831)][[Github](https://github.com/AI4Finance-Foundation/FinGPT)]

- **(CFGPT)** _CFGPT: Chinese Financial Assistant with Large Language Model_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2309.10654)][[Github](https://github.com/TongjiFinLab/CFGPT)][[Model](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full)]

- **(InvestLM)** _InvestLM: A Large Language Model for Investment using Financial Domain Instruction Tuning_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2309.13064)][[Github](https://github.com/AbaciNLP/InvestLM)][[Model](https://huggingface.co/yixuantt/InvestLM-awq)]

- **(DISC-FinLLM)** _DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple Experts Fine-tuning_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2310.15205)][[Github](https://github.com/FudanDISC/DISC-FinLLM)][[Model](https://huggingface.co/Go4miii/DISC-FinLLM)]

- **(FinGPT)** _FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets_ ```Workshop on Instruction Tuning and Instruction Following at NeurIPS 2023```
[[Paper](https://arxiv.org/abs/2310.04793)][[Github](https://github.com/AI4Finance-Foundation/FinGPT)]

- **(FinGPT)** _FinGPT: Democratizing Internet-scale Data for Financial Large Language Models_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2307.10485)][[Github](https://github.com/AI4Finance-Foundation/FinGPT)]

- **(FinMA-ES)** _Dólares or Dollars? Unraveling the Bilingual Prowess of Financial LLMs Between Spanish and English_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2402.07405)][[Github](https://github.com/The-FinAI/PIXIU)]

- **(FinTral)** _FinTral: A Family of GPT-4 Level Multimodal Financial Large Language Models_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2402.10986)]


### Evaluation and Analysis

<h4 id="performance_evaluation_and_analysis_for_popular_financial_tasks">Performance Evaluation and Analysis for Popular Financial Tasks</h4>

- _Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks_ ```EMNLP 2023```
[[Paper](https://arxiv.org/abs/2305.05862)]

- **(PIXIU)** _PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2306.05443)][[Github](https://github.com/The-FinAI/PIXIU)][[Model](https://huggingface.co/ChanceFocus/finma-7b-nlp)]


<h4 id="new_evaluation_frameworks_and_tasks">New Evaluation Frameworks and Tasks</h4>

- **(FinLMEval)** _Is ChatGPT a Financial Expert? Evaluating Language Models on Financial Natural Language Processing_ ```EMNLP 2023```
[[Paper](https://aclanthology.org/2023.findings-emnlp.58/)]

- **(Finben)** _FinBen: A Holistic Financial Benchmark for Large Language Models_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2402.12659)]

- **(FBI)** _Are LLMs Rational Investors? A Study on Detecting and Reducing the Financial Bias in LLMs_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2402.12713)]

- _Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2304.07619)]

- _Can GPT models be Financial Analysts? An Evaluation of ChatGPT and GPT-4 on mock CFA Exams_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2310.08678)][[Github](https://github.com/e-cal/gpt-cfa)]



### LLM-based Methodologies for Financial Tasks and Challenges

<h4 id="confidentiality_and_scarcity_of_high_quality_data">Confidentiality and Scarcity of High-Quality Data</h4>

- _Generating synthetic data in finance: opportunities, challenges and pitfalls_ ```ICAIF 2020```
[[Paper](https://dl.acm.org/doi/10.1145/3383455.3422554)]

- _Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models_ ```ICAIF 2023```
[[Paper](https://dl.acm.org/doi/10.1145/3604237.3626866)][[Github](https://github.com/AI4Finance-Foundation/FinGPT)]

- _Large Language Models as Financial Data Annotators: A Study on Effectiveness and Efficiency_ ```LREC-COLING 2024```
[[Paper](https://aclanthology.org/2024.lrec-main.885/)]

- _No Language is an Island: Unifying Chinese and English in Financial Large Language Models, Instruction Data, and Benchmarks_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2403.06249)][[Github](https://github.com/The-FinAI/PIXIU)]

<h4 id="quantitative_reasoning">Quantitative Reasoning</h4>

- **(ENCORE)** _Enhancing Numerical Reasoning with the Guidance of Reliable Reasoning Processes_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2402.10654)]

- **(PoT)** _Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks_ ```TLMR 2023```
[[Paper](https://arxiv.org/abs/2211.12588)][[Github](https://github.com/wenhuchen/Program-of-Thoughts)]

- **(BRIDGE)** _Exploring Equation as a Better Intermediate Meaning Representation for Numerical Reasoning of Large Language Models_ ```AAAI 2024```
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29879)][[Github](https://github.com/zirui-HIT/Bridge_for_Numerical_Reasoning)]

- **(TAT-LLM)** _TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2401.13223)][[Github](https://github.com/fengbinzhu/TAT-LLM)][[Model](https://huggingface.co/next-tat/tat-llm-7b-fft)]

- _Evaluating LLMs' Mathematical Reasoning in Financial Document Question Answering_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2402.11194)]

<h4 id="multimodal_understanding">Multimodal Understanding</h4>

- **(UReader)** _UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model_ ```EMNLP 2023```
[[Paper](https://aclanthology.org/2023.findings-emnlp.187/)][[Github](https://github.com/LukeForeverYoung/UReader)]

- **(DocLLM)** _DocLLM: A layout-aware generative language model for multimodal document understanding_ ```Arxiv 2023```
[[Paper](https://arxiv.org/abs/2401.00908)][[Github](https://github.com/dswang2011/DocLLM)]

- **(AFIE)** _Enabling and Analyzing How to Efficiently Extract Information from Hybrid Long Documents with LLMs_ ```Arxiv 2024```
[[Paper](https://arxiv.org/abs/2305.16344)]

- **(MANAGER)** _Modal-adaptive Knowledge-enhanced Graph-based Financial Prediction from Monetary Policy Conference Calls with LLM_ ```FinNLP 2024```
[[Paper](https://aclanthology.org/2024.finnlp-1.7/)][[Github](https://github.com/OuyangKun10/MANAGER)]


## Healthcare

### LLMs for Healthcare

- _BioBERT: a pre-trained biomedical language representation model for biomedical text mining_ [[Paper](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)]

- _Clinical BERT: Publicly Available Clinical BERT Embeddings_ [[Paper](https://arxiv.org/abs/1904.03323)]

- _RadBERT: Adapting Transformer-based Language Models to Radiology_ [[Paper](https://pubs.rsna.org/doi/full/10.1148/ryai.210258)]

- _BioMed RoBERTa: Don't Stop Pretraining: Adapt Language Models to Domains and Tasks_ [[Paper](https://arxiv.org/abs/2004.10964)]

- _MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data_ [[Paper](https://arxiv.org/abs/2304.08247)]

- _Me LLaMA: Foundation Large Language Models for Medical Applications_ [[Paper](https://arxiv.org/abs/2402.12749)]

- _LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day_ [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5abcdf8ecdcacba028c6662789194572-Abstract-Datasets_and_Benchmarks.html)]

- _BioInstruct: instruction tuning of large language models for biomedical natural language processing_ [[Paper](https://academic.oup.com/jamia/advance-article-abstract/doi/10.1093/jamia/ocae122/7687618)]

- _Biomedgpt: Open multimodal generative pre-trained transformer for biomedicine_ [[Paper](https://arxiv.org/abs/2308.09442)]

- _MEDITRON-70B: Scaling Medical Pretraining for Large Language Models_ [[Paper](https://arxiv.org/abs/2311.16079)]

- _PMC-LLaMA: toward building open-source language models for medicine_ [[Paper](https://academic.oup.com/jamia/advance-article-abstract/doi/10.1093/jamia/ocae045/7645318)]

- _BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains_ [[Paper](https://arxiv.org/abs/2402.10373)]

- _AlpaCare:Instruction-tuned Large Language Models for Medical Application_ [[Paper](https://arxiv.org/abs/2310.14558)]

- _Clinical Camel: An Open Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding_ [[Paper](https://arxiv.org/abs/2305.12031)]

### Miscellaneous

- _MedEval: A Multi-Level, Multi-Task, and Multi-Domain Medical Benchmark for Language Model Evaluation_ [[Paper](https://arxiv.org/abs/2310.14558)]

- _Robust and Interpretable Medical Image Classifiers via Concept Bottleneck Models_ [[Paper](https://arxiv.org/abs/2310.03182)]

- _The imperative for regulatory oversight of large language models (or generative AI) in healthcare_ [[Paper](https://www.nature.com/articles/s41746-023-00873-0)]

## Law

### LLMs for Law

- _LEGAL-BERT: The muppets straight out of law school_ [[Paper](https://arxiv.org/pdf/2010.02559)]

- _Lawformer: A pre-trained language model for chinese legal long documents_ [[Paper](https://arxiv.org/pdf/2105.03887)]

- _Juru: Legal Brazilian Large Language Model from Reputable Sources_ [[Paper](https://arxiv.org/pdf/2403.18140)]

- _Sabiá-2: A New Generation of Portuguese Large Language Models_ [[Paper](https://arxiv.org/pdf/2403.09887)]

- _Baichuan 2: Open large-scale language models_ [[Paper](https://arxiv.org/pdf/2309.10305)]

- _LawGPT: A Chinese Legal Knowledge-Enhanced Large Language Model_ [[Paper](https://arxiv.org/pdf/2406.04614)]

- _Saullm-7b: A pioneering large language model for law_ [[Paper](https://arxiv.org/pdf/2403.03883)]

- _Chatlaw: Open-source legal large language model with integrated external knowledge bases_ [[Paper](https://arxiv.org/pdf/2306.16092)]

- _Lawyer llama technical report_ [[Paper](https://arxiv.org/pdf/2305.15062)]

### Evaluation

- _LawBench: Benchmarking Legal Knowledge of Large Language Models_ [[Paper](https://arxiv.org/pdf/2309.16289)]

- _LEGALBENCH: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models_ [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/89e44582fd28ddfea1ea4dcb0ebbf4b0-Paper-Datasets_and_Benchmarks.pdf)]

- _LAiW: A Chinese Legal Large Language Models Benchmark_ [[Paper](https://arxiv.org/pdf/2310.05620)]

## Ethics

### Ethics in LLM + Finance

- _Deficiency of Large Language Models in Finance: An Empirical Examination of Hallucination_ ```ICBINB 2023``` [[Paper](https://openreview.net/forum?id=SGiQxu8zFL)]

- _Hallucination-minimized Data-to-answer Framework for Financial Decision-makers_ ```BigData 2023``` [[Paper](https://arxiv.org/pdf/2311.07592)]

- _Beavertails: Towards Improved Safety Alignment of LLM via a Human-preference Dataset_ ```NeurIPS 2024``` [[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/4dbb61cb68671edc4ca3712d70083b9f-Paper-Datasets_and_Benchmarks.pdf)] 

### Ethics in LLM + Healthcare

- _Large Language Models in Medicine_ ```Nature medicine 2023``` [[Paper](https://drive.google.com/file/d/1FKEGsSZ9GYOeToeKpxB4m3atGRbC-TSm/view)]

- _Ethics of Large Language Models in Medicine and Medical Research_ ```The Lancet Digital Health 2023``` [[Paper](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(23)00083-3/fulltext)]

- _AI and Ethics: A Systematic Review of the Ethical Considerations of Large Language Model Use in Surgery Research_ ```Healthcare 2024``` [[Paper](https://www.mdpi.com/2227-9032/12/8/825)]

- _The Ethics of ChatGPT in Medicine and Healthcare: A Systematic Review on Large Language Models (LLMs)_ ```npj Digital Medicine``` [[Paper](https://www.nature.com/articles/s41746-024-01157-x)]

- _Embracing Large Language Models for Medical Applications: Opportunities and Challenges_ ```Cureus 2023``` [[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10292051/)]

- _The Challenges for Regulating Medical Use of ChatGPT and Other Large Language Models_ ```JAMA 2023``` [[Paper](https://jamanetwork.com/journals/jama/article-abstract/2807167)]

- _Leveraging Generative AI and Large Language Models: a Comprehensive Roadmap for Healthcare Integration_ ```Healthcare 2023``` [[Paper](https://www.mdpi.com/2227-9032/11/20/2776)]

- _Challenges and Barriers of Using Large Language Models (LLM) such as ChatGPT for Diagnostic Medicine with a Focus on Digital Pathology–a Recent Scoping Review_ ```Diagnostic pathology 2024``` [[Paper](https://link.springer.com/article/10.1186/s13000-024-01464-7)]

- _Ethical Dilemmas, Mental Health, Artificial Intelligence, and LLM-based Chatbots_ ```IWBBIO 2023``` [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-34960-7_22)]


### Ethics in LLM + Law

- _Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models_ ```Journal of Legal Analysis 2024``` [[Paper](https://academic.oup.com/jla/article-abstract/16/1/64/7699227)]

- _I Am Not a Lawyer, But...: Engaging Legal Experts towards Responsible LLM Policies for Legal Advice_ ```ACM FAccT 2024``` [[Paper](https://dl.acm.org/doi/pdf/10.1145/3630106.3659048)]

- _Evaluation Ethics of LLMs in Legal Domain_ ```ArXiv preprint 2024``` [[Paper](https://arxiv.org/pdf/2403.11152)]
