from flask import Flask, render_template, request, jsonify
import requests
import json
from openai import OpenAI
import numpy as np
import itertools
import re
from Levenshtein import distance
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from py2neo import Graph, Subgraph
from py2neo import Node, Relationship, Path
from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
graph = Graph('http://localhost:7474/', name='neo4j', password='136339fkm')
def llm(proset,text):
    client = OpenAI(
        api_key = "sk-ftCFLyV6YJOlSdXJ6q5fTbGwMmxXwuRYQlim9lqJ8OE9C1OS",
        base_url = "https://api.moonshot.cn/v1",
    )
    completion = client.chat.completions.create(
        model = "moonshot-v1-32k",
        messages = [
            {"role": "system", "content": proset},
            {"role": "user", "content": text}
        ],
        tools = [{
            "type": "function",
            "function": {
                "name": "CodeRunner",
                "description": "代码执行器，支持运行 python 和 javascript 代码",
                "parameters": {
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript"]
                        },
                        "code": {
                            "type": "string",
                            "description": "代码写在这里"
                        }
                    },
                "type": "object"
                }
            }
        }],
        temperature = 0.3,
    )
    return completion.choices[0].message.content

# 加载医学词典和停用词库
jieba.load_userdict('static/data/医学词典.txt')
jieba.load_userdict('static/data/中医药词表.txt')
stop_words = open('static/data/stopwords.txt','r+',encoding='utf-8').read().split('\n')+['所致','证见','伴有','症状']  # 示例停用词
def pipei(li1,li2):
    dis={}
    for a in li2:
        dis[a]=[]
        for b in li1:
            # a=re.sub(u"\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "",a)
            # b=re.sub(u"\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "",b)
            dis[a]+=[(max(len(a),len(b))-distance(a,b))/(max(len(a),len(b)))]
    return pd.DataFrame(dis,index=li1)
def process_text(text):
    words = jieba.lcut(str(text))
    return ' '.join([word for word in words if word not in stop_words])
def process_nostop(text):
    words = jieba.lcut(str(text))
    return ' '.join(words)
def sdscore(st1,li1):
    vectorizer = TfidfVectorizer()
    li=[process_nostop(n) for n in li1]
    st=process_nostop(st1)
    symptoms_vectors = vectorizer.fit_transform(li)
    user_vector = vectorizer.transform([st])
    return ((cosine_similarity(user_vector, symptoms_vectors)+np.array(pipei(li,[st])[st]))/2).flatten(),pd.DataFrame({"字段":li1,"分词后":li,'cos_sim':cosine_similarity(user_vector, symptoms_vectors).flatten(),'distance':np.array(pipei(li,[st])[st]).flatten()})
def medname(text):
    try:
        return re.search("通用名称：(.*?)\n",str(text)).group(1)
    except:
        return re.search("通用名称：(.*?)$",str(text)).group(1)
meg_data=pd.read_excel("static/data/药品通说明书数据（8803条）.xlsx")
meg_data_orig=pd.read_excel("static/data/药品通说明书数据（8803条）.xlsx")
def nameget(var,st):
    try:
        return re.search(st+r"：(.*?)$",var,re.MULTILINE).group(1)
    except:
        return ""
for i in ["通用名称","商品名称","英文名称","汉语拼音"]:
    meg_data_orig[i]=meg_data_orig['药品名称'].apply(nameget,args=(i,))
def getvar(df,var,medname):
    x=list(df[df['通用名称'] == medname][var])[0]
    if type(x) == str and len(x)>0 and x==x:
       return x.strip()
    else:
        return "暂无"
meg_data["通用名称"]=meg_data["药品名称"].apply(medname)
meg_data=meg_data.drop_duplicates(subset=["通用名称"]).reset_index(drop=True)
effects_df=meg_data[["通用名称","适应症"]]
meg_data=meg_data.drop_duplicates(subset=['通用名称']).set_index("通用名称")
yuju=''
for i in meg_data.columns:
    yuju+="'["+i+"]'+meg_data['"+i+"'].astype(str)+"
yuju=yuju.strip("+")
meg_data['信息']=eval(yuju)
meg_data=meg_data['信息']
effects_df['适应症_p'] =effects_df['适应症'].apply(process_text)
data=pd.read_csv("static/data/药品通数据_知识图谱.csv",encoding="ANSI")
data["成分明细"]=data["成分明细"].str.replace("±"," ")
def query_med_get(text):
    query="""
    ###任务介绍
    你需要从句子中抽取出“药品”的名称以及别名。
    下面给出提取“药品”的名称以及别名的3个示例:
    句子1:感冒了，可以白加黑和妈咪爱一吃吗
    输出1:{"白加黑","妈咪爱","氨麻苯美片","氨酚伪麻美芬片Ⅱ","枯草杆菌二联活菌颗粒","枯草杆菌、肠球菌二联活菌多维颗粒剂"}
    句子2:眼镜酸涩可以用沙普爱丝吗？
    输出2:{"莎普爱思","苄达赖氨酸滴眼液"}
    句子3:有什么止痛片可以吃？
    输出3:{}
    ###输出要求:
    请从给定的句子中抽取出所有“药品”的名称以及别名。要求:(1)输出的每个结果格式为:{"x","y","z"}，若无成分实体则直接输出{}。(2)输出的每个结果用","分割，如:{"去痛片","索密痛"}。(3)请按照规定格式输出，不要添加其他内容。
    ###句子:
    """
    return llm("You are a veteran pharmacist.",query+text)
def query_sympt_get(text):
    query="""
    ###任务介绍
    你需要从句子中抽取出“症状或功效”的表达。
    下面给出提取“症状或功效”的表达的3个示例:
    句子1:流感干咳吃什么药
    输出1:《流感干咳》
    句子2:既有鼻炎又有咽炎吃什么消炎药好
    输出2:《鼻炎，咽炎》
    句子3:男人助阳补肾吃什么好？
    输出3:《助阳补肾》
    ###输出要求:
    请从给定的句子中抽取出所有“症状或功效”的表述。要求:(1)输出的结果格式为:《x，y》，若无症状或功效则直接输出《》。(2)输出的每个结果用"，"分割。(3)请按照规定格式输出，不要添加其他内容。
    ###句子:
    """
    return llm("You are a Senior doctor.",query+text)
def query_med_repl(text,medli):
    query="""
    ###任务介绍
    你需要从mset集合里面的药品名称替代query里的相应的药品名称或别名，并用专业语言润色输出query。
    下面给出替换药品的3个示例:
    query1:感冒了，可以白加黑和妈咪爱一吃吗
    mset1:{"氨麻苯美片","氨酚伪麻美芬片Ⅱ","枯草杆菌二联活菌颗粒","枯草杆菌、肠球菌二联活菌多维颗粒剂"}
    输出1:《感冒了，氨麻苯美片和枯草杆菌二联活菌颗粒能一起吃吗？》
    query2:眼镜酸涩可以用苄达赖氨酸滴眼液吗？
    mset2:{"阿莫西林克拉维酸钾分散片","复方磺胺甲噁唑钠滴眼液","苄达赖氨酸滴眼液","克拉霉素片"}
    输出2:《眼睛感到酸涩时，是否可以使用苄达赖氨酸滴眼液呢？》
    query3:臣功再欣和感康哪个好用？
    mset3:{"复方锌布颗粒剂","馥感啉口服液","养阴口香合剂"}
    输出3:《复方锌布颗粒剂和感康哪个好用？》
    ###输出要求:
    请在给定的句子中替换相应的药品名称。要求:(1)若问题中无药品则无需替换。(2)输出的问题用《》括起来。(3)请按照规定格式输出，不要添加其他内容。(4)若mset中无query中相应药品则无需替换。
    ###
    query:"""
    return llm("You are a veteran pharmacist.",query+text+"\n mset:"+str(medli))
def query_type(text):
    query="""
    ###任务介绍
    你需要判断句子属于以下哪种需求类型：
    a:提供药品名称，查找药品作用功效，是否对症，用法用量或注意事项。
    b:提供症状，查找适用药物。
    c:提供药品但未提供症状查找替代药物或相似药物。
    d:提供多个药品名称，进行药品比较。
    下面给出问题分类的8个示例:
    句子1:吃法罗培南17天能停药吗
    输出1:《a》
    句子2:眼镜酸涩可以用沙普爱丝吗？
    输出2:《a》
    句子3:邱医生您好，想问您下肺癌患者在有炎症（如感冒，咳嗽等）时除了拜复乐之外还有什么替代的药比较好，拜复乐副作用比较大，谢谢（男，32岁）
    输出3:《b》
    句子4:什么药可以替代双氯芬酸钠（男，62岁）
    输出4:《c》
    句子5:女33岁，之前有过胃病，在50天前做过胃镜胃镜显示糜烂性胃炎，之前也是上火吃东西就吐两天以后就会自愈，就在4天前上火了又开始吐，一直到今天现在严重了喝水都吐，喝水立马就会吐出来，现在胃里疼，烧热，我需要吃什么药物缓解。感谢医生（女，33岁）
    输出5:《b》
    句子6:依托度酸和来氟米特哪个副作用小？
    输出6:《d》
    句子7:依托度酸和来氟米特的副作用分别有什么？
    输出7:《a》
    句子8:阿魏酸哌可以替代阿司匹林肠溶片吗？
    输出8:《d》
    ###输出要求:
    请将给定的句子分为《a》，《b》，《c》，《d》三类。要求:(1)输出的每个结果要用《》括起来，格式为:《x》。(2)请按照规定格式输出，不要添加其他内容。
    ###句子:
    """
    return llm("You are a Senior doctor.",query+text)
def med_search(text,massage=''):
    query="""
    ###任务介绍
    你正在医院接诊，患者来咨询药品相关信息，你需要根据给出的药品信息详细回答患者的问题。下面是2个医患真实对话示例，你可以作为参考。
    提问1:马来酸左氨氯地平片能服半片吗
    回答1:马来酸左氨氯地平片可以服半片，但不建议患者这样服用。因为该药物的服用方法是一次2.5mg，一日1次，如果服用半片，可能会导致血药浓度过高，从而增加发生不良反应的风险。马来酸左氨氯地平片属于长效的钙离子拮抗剂，可以抑制钙离子内流，扩张血管，使外周阻力降低，从而起到降压的作用。该药物的服用方法是一次2.5mg，一日1次，如果患者服用半片，可能会导致血药浓度过高，从而引起反射性心率增快、面部潮红、下肢水肿等不良反应。同时还可能会导致患者出现头痛、牙痛、肌肉痉挛等现象。因此，不建议患者这样服用该药物。另外，建议患者在医生的指导下服用该药物，避免自行服用，以免因用药不当引起不适症状。如果患者在服用该药物期间出现不适症状，建议及时就医治疗。
    提问2:高血脂吃辛伐他汀和血塞通胶囊有用吗
    回答2:高血脂吃辛伐他汀和血塞通胶囊有用，但需要在医生的指导下进行用药治疗。辛伐他汀属于西药，具有降低血脂的功效，而血塞通胶囊属于中成药，具有活血化瘀的功效。1、辛伐他汀辛伐他汀属于西药，主要成分为辛伐他汀钙，可以用于治疗高脂血症、冠状动脉粥样硬化性心脏病等疾病引起的不适症状，比如头晕、胸闷等。如果患者存在高血脂的情况，可能会使血液黏稠度增加，从而出现头晕、胸闷等不适症状，此时可以在医生的指导下服用辛伐他汀进行治疗，能够有效改善病情。2、血塞通胶囊血塞通胶囊主要成分为三七总皂苷，具有活血祛瘀、通脉活络的功效，在临床可以用于治疗中风偏瘫、胸痹心痛、脑血管病后遗症等症状。如果患者存在高血脂的情况，可能会导致血液黏稠度增加，从而出现头晕、胸闷等不适症状，此时可以遵医嘱服用血塞通胶囊进行治疗，在一定程度上可以缓解患者的症状。在服用药物期间要注意饮食清淡，可以多吃膳食纤维丰富的水果和蔬菜，比如苹果、芹菜等，避免吃高脂肪、高热量食物，以免影响药物吸收。如果服用药物后出现不适症状，建议患者及时就医治疗。
    ###输出要求:
    (1)以亲切、尊重的方式称呼患者，语言通俗易懂且具有专业性。(2)回答患者关于药品的查询问题，提供准确的药品信息，并结合患者的具体情况给出建议。(3)注意患者问题中的疾病史和家族病史。(4)明确患者问题中真实需求，关注患者的感受、价值观、态度、同理心等内容。(5)给出原理和原因，全面告知每种药品优劣，给出建议。(6)考虑患者身心及经济因素，表达鼓励与信心。
    ###患者问题:
    """
    if massage=='':
        return llm("You are a veteran pharmacist.",query+text)
    else:
        return llm("You are a veteran pharmacist.","\n药品信息:\n"+massage+'\n请参考以上信息完成以下任务:\n'+query+text)
def med_compare(text,massage=''):
    query="""
    ###任务介绍
    你正在医院接诊，患者想对多种药品进行比较以选择更适合自己的药物。下面是2个医患真实对话示例，你可以作为参考。
    提问1:去痛片和布洛芬哪个好
    回答1:去痛片和布洛芬都属于非甾体抗炎药，难以比较哪个更好，具体选择哪种药物还需根据患者的具体情况来决定。建议在医生指导下使用。1、去痛片：主要成分为氨基比林和非那西丁，适用于缓解轻至中度的各种疼痛，如头痛、牙痛、神经痛等。由于其含有阿司匹林成分，因此不适宜用于胃溃疡或十二指肠溃疡的治疗。2、布洛芬：是一种常见的解热镇痛类非处方药，常用于缓解普通感冒或流行性感冒引起的发热症状以及偏头痛、关节痛等症状。该药物对消化道黏膜有一定的刺激作用，在长期大量用药时可能会引起恶心呕吐、腹胀腹泻等不良反应。对于有严重肝肾功能损害或其他严重疾病的人群，应避免自行服用上述任何一种止痛药物，并及时就医寻求专业医师的帮助与指导。同时，应注意保持良好的生活习惯，合理饮食，适量运动，以减少不适的发生。
    提问2:阿魏酸哌可以替代阿司匹林肠溶片吗
    回答2:由于阿司匹林肠溶片和阿魏酸哌在功效和适用人群上存在差异，因此不能直接替代使用。阿司匹林肠溶片是一种非甾体抗炎药，主要通过抑制环氧合酶，减少前列腺素的合成来发挥其作用。而阿魏酸哌是一种天然的抗氧化剂和自由基清除剂，具有抗炎、抗氧化和保护血管的作用。两者在功效和适用人群上存在差异，因此不能直接替代。阿司匹林肠溶片主要用于预防心血管疾病和血栓形成，适用于有心血管疾病风险的人群。而阿魏酸哌则主要用于改善血液循环、降低血脂和血糖，适用于有代谢综合征或糖尿病的人群。对于需要使用阿司匹林肠溶片的人群，应按照医生的建议进行治疗，并定期进行复查。而对于需要改善代谢状况的人群，则可以考虑使用阿魏酸哌或其他适合的药物。
    ###输出要求:
    (1)以亲切、尊重的方式称呼患者，语言通俗易懂且具有专业性。(2)首先明确给出建议或说明无法简单判定哪种药品更好，需综合多方面因素考量。(3)对比药品的疗效、价格、安全性等关键信息，详细阐述各自的特点和优势劣势。(4)结合患者可能存在的身体状况、疾病史、用药史等情况给出针对性建议，体现对患者个体差异的关注。(5)用通俗易懂的语言解释专业术语，确保患者能够理解比较结果。(6)表达愿意进一步为患者解答疑问的态度，增强患者的信任感。
    ###患者问题:
    """
    if massage=='':
        return llm("You are a veteran pharmacist.",query+text)
    else:
        return llm("You are a veteran pharmacist.","\n药品信息:\n"+massage+'\n请参考以上信息完成以下任务:\n'+query+text)
def med_advice(text,massage=''):
    query="""
    ###任务介绍
    你正在医院接诊，患者根据自身症状来咨询适合的药物。下面是2个医患真实对话示例，你可以作为参考。
    提问1:脚气用什么药物治疗比较好
    回答1:脚气可以使用硝酸咪康唑乳膏、盐酸特比萘芬喷雾剂、氟康唑胶囊等药物进行治疗是比较好的。1.硝酸咪康唑乳膏该药物适用于真菌感染引起的脚气，具有抗菌和抗真菌的作用。2.盐酸特比萘芬喷雾剂该药物适用于皮肤真菌感染，包括脚气，能有效抑制真菌生长。3.氟康唑胶囊对于酵母菌和某些真菌感染，如脚气，氟康唑胶囊有良好的治疗效果。在治疗脚气时，应根据医生建议选择合适的药物，并严格按照药品说明书或医嘱使用。患者应注意保持足部清洁干燥，避免穿潮湿的鞋袜，并注意个人卫生。
    提问2:小儿化痰止咳吃什么药最有效
    回答2:小儿化痰止咳可以考虑使用盐酸氨溴索、小儿止嗽化痰颗粒、氨溴特罗口服溶液口服溶液等药物进行治疗。1.盐酸氨溴索使用该药物的原因是小儿咳嗽时痰液黏稠，难以咳出，而盐酸氨溴索可以增加呼吸道黏膜浆液腺的分泌，减少黏液腺的分泌，从而稀释痰液，使其易于咳出。2.小儿止嗽化痰颗粒使用该药物的原因是小儿咳嗽伴有痰多、不易咳出时，而该药物具有清热化痰、止咳平喘的功效，适用于小儿肺热咳嗽。3.氨溴特罗口服溶液口服溶液使用该药物的原因是小儿咳嗽伴有喘息时，而氨溴特罗口服溶液口服溶液可以同时扩张支气管和稀释痰液，缓解喘息和咳嗽症状。在治疗小儿化痰止咳时，应遵医嘱选择合适的药物，并注意观察症状变化。同时，保持室内空气流通、湿度适宜，避免吸入刺激性气体和烟雾。
    ###输出要求:
    (1)以亲切、尊重的方式称呼患者，语言通俗易懂且具有专业性。(2)根据患者提供的用药需求，为患者提供综合的用药建议。(3)对比推荐药品的关键信息，详细阐述各自的特点和优势劣势。(4)结合患者可能存在的身体状况、过敏成分、剂型偏好等情况给出针对性建议，体现对患者个体差异的关注。(5)明确患者问题中真实需求，关注患者的感受、价值观、态度、同理心等内容。(6)表达愿意进一步为患者解答疑问的态度，增强患者的信任感。
    ###患者问题:
    """
    if massage=='':
        return llm("You are a Senior doctor.",query+text)
    else:
        return llm("You are a Senior doctor.","\n药品信息:\n"+massage+'\n请参考以上信息完成以下任务:\n'+query+text)
def med_replace(text,massage=''):
    query="""
    ###任务介绍
    你正在医院接诊，患者因特殊情况需要寻找药品替代方案。下面是2个医患真实对话示例，你可以作为参考。
    提问1:托度酸胶囊可以用什么药代替
    回答1:通常情况下，依托度酸胶囊可以用塞来昔布胶囊、双氯芬酸钠缓释片、布洛芬缓释胶囊、吲哚美辛胶囊、萘普生胶囊等药物进行代替，患者需要在医生指导下用药。1、塞来昔布胶囊塞来昔布胶囊属于非甾体抗炎药，具有抗炎、镇痛的作用，在临床上可以用于治疗骨关节炎、风湿性关节炎等疾病引起的疼痛症状。如果患者出现上述症状，可以在医生指导下使用塞来昔布胶囊进行治疗。2、双氯芬酸钠缓释片双氯芬酸钠缓释片属于非甾体抗炎药，具有抗炎、镇痛的作用，在临床上可以用于治疗急性关节炎症和痛风发作、慢性关节炎症、强直性脊柱炎等疾病引起的疼痛症状。如果患者出现上述症状，可以在医生指导下使用双氯芬酸钠缓释片进行治疗。3、布洛芬缓释胶囊布洛芬缓释胶囊属于非甾体抗炎药，具有解热镇痛的作用，在临床上可以用于治疗偏头痛、痛经、手术后疼痛等症状。如果患者出现上述症状，可以在医生指导下使用布洛芬缓释胶囊进行治疗。4、吲哚美辛胶囊吲哚美辛胶囊也属于非甾体抗炎药，具有解热镇痛的作用，在临床上可以用于治疗关节炎、软组织损伤、偏头痛等症状。如果患者出现上述症状，可以在医生指导下使用吲哚美辛胶囊进行治疗。5、萘普生胶囊萘普生胶囊是一种非甾体抗炎药，具有解热镇痛的作用，在临床上可以用于治疗关节炎、软组织损伤、偏头痛等症状。如果患者出现上述症状，可以在医生指导下使用萘普生胶囊进行治疗。建议患者在医生指导下使用药物，避免自行用药，以免引起不良反应，导致病情加重。同时，在用药期间饮食上尽量以清淡易消化、营养均衡为主，避免进食辛辣、生冷、油腻等刺激性食物，以免加重不适症状。另外，还可以适当进行运动，如打羽毛球、慢跑等，能辅助增强机体免疫力。若期间出现明显不适，还需及时就医诊治，以免延误病情。
    提问2:有没有代替小儿氨酚那敏的中成药，儿童3岁
    回答2:小儿感冒颗粒、清开灵颗粒是治疗小孩感冒的中成药。婴儿感冒是常有的事，需要及时应用药物治疗感冒。治疗感冒的药物有很多，比如小儿氨酚黄那敏颗粒和小儿氨酚烷胺颗粒，是常用的西药。还有很多中成药，如小儿感冒颗粒、清开灵颗粒等，有基本一样的效果，都是针对小儿感冒症状的药物。小儿感冒颗粒是一种中成药，主要由菊花、连翘、地黄、薄荷等成分组成，具有清热解毒、祛风解表等功效，对治疗风寒感冒引起的头痛、咳嗽、鼻塞等症状有很好的效果，这种药需要用开水冲服，具体剂量要参考说明书或遵医嘱使用。清开灵颗粒是一种中成药，由胆酸、黄芩苷、金银花等组成，具有清热解毒、安神定志的功效。具体药物的选择需要根据孩子的具体症状和接受程度来给药。
    ###输出要求:
    (1)以亲切、尊重的方式称呼患者，语言通俗易懂且具有专业性。(2)根据患者的具体情况，推荐可能的替代药物，并详细介绍替代药物的作用、适用症状、可能的不良反应等。(3)注意患者问题中的疾病史和家族病史。(4)注意患者真实需求和感受、价值观、态度、同理心等内容。(5)给出原理和原因，全面告知推荐药品优劣，给出建议。(6)考虑患者身心及经济因素，表达鼓励与信心
    ###患者问题:
    """
    if massage=='':
        return llm("You are a Senior doctor.",query+text)
    else:
        return llm("You are a Senior doctor.","\n药品信息:\n"+massage+'\n请参考以上信息完成以下任务:\n'+query+text)
import pandas as pd
import json
def initial(graph):
    graph = Graph('http://localhost:7474/', name='neo4j', password='136339fkm')
    data=pd.read_csv("static/data/药品通数据_知识图谱.csv",encoding="ANSI")
    data = data[data["功效或适应症"] != '{}']
    data["成分明细"]=data["成分明细"].str.replace("±"," ")
    graph.run('match (n) detach delete n')
    nodes={}
    for i in data.index:
        if data["指导价"][i]==data["指导价"][i]:
            if data["商品名称"][i]==data["商品名称"][i]:
                nodes["药品:"+data["通用名称"][i]+":"+str(data["指导价"][i])] = Node("药品",name = data["通用名称"][i],指导价=data["指导价"][i],商品名称=data["商品名称"][i])
            else:
                nodes["药品:"+data["通用名称"][i]+":"+str(data["指导价"][i])] = Node("药品",name = data["通用名称"][i],指导价=data["指导价"][i])
        else:
            if data["商品名称"][i]==data["商品名称"][i]:
                nodes["药品:"+data["通用名称"][i]] = Node("药品",name = data["通用名称"][i],商品名称=data["商品名称"][i])
            else:
                nodes["药品:"+data["通用名称"][i]] = Node("药品",name = data["通用名称"][i])
    for i in data["用法"]:
        for ii in eval(i):
            nodes["用法:"+ii]=Node("用法",name = ii)
    for i in data["剂型"]:
        for ii in eval(i):
            nodes["剂型:"+ii]=Node("剂型",name = ii)
    for i in data["成分明细"]:
        for ii in eval(i):
            nodes["成分:"+ii]=Node("成分",name = ii)
    for i in data["功效或适应症"]:
        for ii in eval(i):
            nodes["适应症:"+ii]=Node("适应症",name = ii)
    for i in data["特殊人群"]:
        for ii in eval(i)["忌用人群"]:
            nodes["特殊人群:"+ii] = Node("特殊人群",name = ii)
        for ii in eval(i)["禁用人群"]:
            nodes["特殊人群:"+ii] = Node("特殊人群",name = ii)
        for ii in eval(i)["慎用人群"]:
            nodes["特殊人群:"+ii] = Node("特殊人群",name = ii)
    relas=[]
    for i in data.index:
        if data["指导价"][i]==data["指导价"][i]:
            node_1 = nodes["药品:"+data["通用名称"][i]+":"+str(data["指导价"][i])]
        else:
            node_1 = nodes["药品:"+data["通用名称"][i]]
        for ii in eval(data["用法"][i]):
            node_2 = nodes["用法:"+ii]
            relas+=[Relationship(node_1, "使用途径", node_2)]
        for ii in eval(data["剂型"][i]):
            node_2 = nodes["剂型:"+ii]
            relas+=[Relationship(node_1, "采用剂型", node_2)]
        for ii in eval(data["成分明细"][i]):
            node_2 = nodes["成分:"+ii]
            relas+=[Relationship(node_1, "包含成分", node_2)]
        for ii in eval(data["功效或适应症"][i]):
            node_2 = nodes["适应症:"+ii]
            relas+=[Relationship(node_1, "功能主治", node_2)]
        for ii in eval(data["特殊人群"][i])["忌用人群"]:
            node_2 = nodes["特殊人群:"+ii]
            relas+=[Relationship(node_1, "忌用人群", node_2)]
        for ii in eval(data["特殊人群"][i])["禁用人群"]:
            node_2 = nodes["特殊人群:"+ii]
            relas+=[Relationship(node_1, "禁用人群", node_2)]
        for ii in eval(data["特殊人群"][i])["慎用人群"]:
            node_2 = nodes["特殊人群:"+ii]
            relas+=[Relationship(node_1, "慎用人群", node_2)]
    subgraph = Subgraph(list(nodes.values()), relas)
    tx = graph.begin()
    tx.create(subgraph)
    graph.commit(tx)
    return graph
graph=initial(graph)
def query_analy(t):
    text="""
###任务介绍
你需要从句子中抽取患者的用药需求，其中包括["适应症","剂型","用法","成分","特殊人群"]五个维度，每个维度都要提取出需要和不需要两个维度。
下面给出提取用药需求的3个示例:
句子1:我36岁正在备孕，患有糖尿病且对杏仁过敏，这几天突然感觉喉咙痛，经常咳嗽还有浓痰，请问我可以吃点什么药，最好是冲服的，之前吃药片胶囊有些咽不下去。
输出1:{"适应症":{"需要":["咽痛","咳嗽","浓痰"],"不需要":[]},"剂型":{"需要":[],"不需要":["片剂","胶囊"]},"用法":{"需要":["冲服"],"不需要":[]},"成分":{"需要":[],"不需要":["杏仁"]},"特殊人群":{"需要":[],"不需要":["孕妇","糖尿病患者"]}}
句子2:胃老是胀气。一天到晚的老是放屁。饿过头呢也会打嗝。一天两三次肚子痛想大便。偶尔拉稀。大部分都是啦的少量淡黄色大便。晚上多吃呢一点饭菜，或者多喝呢很多水或者酒。容易反胃呕吐。或者第二天早上胃部胀痛难忍。连续不间断的打嗝。第二天早上胀痛平均一月一两次。。放屁打嗝每天都会有。由于当初在医院检查医生说普遍情况以为没太大问题，就没开药。这次希望医生可以根据症状判断并帮助开药（男，76岁）
输出2:{"适应症":{"需要":["胃腹胀满","大便溏泻","嗳气","呕吐"],"不需要":[]},"剂型":{"需要":[],"不需要":[]},"用法":{"需要":[],"不需要":[]},"成分":{"需要":[],"不需要":[},"特殊人群":{"需要":[],"不需要":["老人"]}}
句子3:我一直有高血压，之前尿尿会疼，医生说是尿道感染，但没给开药，我应该用什么药呢，最好是直接用到患处的，应该起效会快一点
输出3:{"适应症":{"需要":["尿道感染"],"不需要":[]},"剂型":{"需要":[],"不需要":[]},"用法":{"需要":["尿道给药"],"不需要":[]},"成分":{"需要":[],"不需要":[},"特殊人群":{"需要":[],"不需要":["高血压患者"]}}
###输出要求:
请从给定的句子中抽取出所有用药需求。要求:
(1)输出格式为：{"适应症":{"需要":[],"不需要":[]},"剂型":{"需要":[],"不需要":[]},"用法":{"需要":[],"不需要":[]},"成分":{"需要":[],"不需要":[]},"特殊人群":{"需要":[],"不需要":[]}}。
(2)用法字段必须在以下列表中选取（若无适合则无需选取，禁止填写列表之外的需求）：['皮内注射', '栓剂', '吸入', '肌肉注射', '静脉滴注', '皮下注射', '滴眼', '泡服', '静脉注射', '含漱', '滴鼻', '尿道给药', '直肠给药', '嚼服', '医疗器械', '阴道给药', '滴耳', '局部给药', '外用', '冲服', '加温软化', '口服', '注射', '口腔给药', '喷雾', '含服', '煎服']
剂型字段必须在以下列表中选取（若无适合则无需选取，禁止填写列表之外的需求）：['栓剂', '片剂', '中药剂型', '喷雾剂', '气雾剂', '植入剂', '注射剂', '颗粒剂', '胶囊剂', '软膏剂', '液体剂型', '医疗器械', '粉剂', '贴剂', '膜剂']
其他字段尽量以专业医学词汇表述。
(3)非主诉症状应放入特殊人群字段中。
(4)若某字段未被提到，也要写"字段":{"需要":[],"不需要":[]}。
(5)注意年龄，若年龄符合老人或婴儿等也要写上。
(6)请按照规定格式输出，不要添加其他内容。
(7)不要猜测患者的用药需求，若需求在句子中未明确提出则不可写入输出中。
###句子:
    """
    return re.search("\{(.*)\}",llm("you are a senior doctor.",text+str(t)),re.S).group(0)
def query_analy_sub(t):
    text="""
###任务介绍
你需要从句子中抽取患者的替换药品需求，其中包括["适应症","剂型","用法","成分","特殊人群"]五个维度，每个维度都要提取出需要和不需要两个维度。
下面给出提取用药需求的3个示例:
句子1:我36岁正在备孕，患有糖尿病且对杏仁过敏，我可以吃点什么药替代快克，最好是冲服的，之前吃药片胶囊有些咽不下去。
输出1:{"适应症":{"需要":[],"不需要":[]},"剂型":{"需要":[],"不需要":["片剂","胶囊"]},"用法":{"需要":["冲服"],"不需要":[]},"成分":{"需要":[],"不需要":["杏仁"]},"特殊人群":{"需要":[],"不需要":["孕妇","糖尿病患者"]}}
句子2:当初在医院检查医生说推荐吃氟哌酸，但是没太大问题，就没开药。这次希望医生可以根据症状判断并帮助开药（男，76岁）
输出2:{"适应症":{"需要":[],"不需要":[]},"剂型":{"需要":[],"不需要":[]},"用法":{"需要":[],"不需要":[]},"成分":{"需要":[],"不需要":[},"特殊人群":{"需要":[],"不需要":["老人"]}}
句子3:我一直有高血压，医生给开了阿莫西林胶囊，我现在药店找不到了，能用什么药替换呢，最好是直接涂抹的，应该起效会快一点
输出3:{"适应症":{"需要":[],"不需要":[]},"剂型":{"需要":["软膏剂"],"不需要":[]},"用法":{"需要":["外用"],"不需要":[]},"成分":{"需要":[],"不需要":[},"特殊人群":{"需要":[],"不需要":["高血压患者"]}}
###输出要求:
请从给定的句子中抽取出所有用药需求。要求:
(1)输出格式为：{"适应症":{"需要":[],"不需要":[]},"剂型":{"需要":[],"不需要":[]},"用法":{"需要":[],"不需要":[]},"成分":{"需要":[],"不需要":[]},"特殊人群":{"需要":[],"不需要":[]}}。
(2)用法字段必须在以下列表中选取（若无适合则无需填写）：['皮内注射', '栓剂', '吸入', '肌肉注射', '静脉滴注', '皮下注射', '滴眼', '泡服', '静脉注射', '含漱', '滴鼻', '尿道给药', '直肠给药', '嚼服', '医疗器械', '阴道给药', '滴耳', '局部给药', '外用', '冲服', '加温软化', '口服', '注射', '口腔给药', '喷雾', '含服', '煎服']
剂型字段必须在以下列表中选取（若无适合则无需填写）：['栓剂', '片剂', '中药剂型', '喷雾剂', '气雾剂', '植入剂', '注射剂', '颗粒剂', '胶囊剂', '软膏剂', '液体剂型', '医疗器械', '粉剂', '贴剂', '膜剂']
其他字段尽量以专业医学词汇表述。
(3)非主诉症状应放入特殊人群字段中。
(4)若某字段未被提到，也要写"字段":{"需要":[],"不需要":[]}。
(5)注意年龄，若年龄符合老人或婴儿等也要写上。
(6)请按照规定格式输出，不要添加其他内容。
(7)不要猜测患者的用药需求，若需求在句子中未明确提出则不可写入输出中。
###句子:
    """
    return re.search("\{(.*)\}",llm("you are a senior doctor.",text+str(t)),re.S).group(0)

def node_di(df,se):
    di={}
    for r in list(se):
        for i in df[r]:
            i=i["name"]
            di[i]=di.get(i,0)+1
    df1 = pd.DataFrame(sorted(di.items(), key=lambda x: x[1] ,reverse=True),columns=['name','count'])
    return list(df1['name'])
def rela_di(df,se):
    di={}
    for r in list(se):
        for i in df[r]:
            di[(i.start_node['name'],i.end_node['name'])]=di.get((i.start_node['name'],i.end_node['name']),0)+1
    return list(di.keys())
def node_edge(m_li,s_li,x_li):
    df_x=pd.DataFrame(x_li,columns=["药品","适应症"])
    nodes=[]
    iddi={}
    a=0
    for i in m_li:
        a+=1
        nodes+=[{'id':str(a),'type':'m','name':i}]
        iddi[i]=a
    for i in s_li:
        a+=1
        nodes+=[{'id':str(a),'type':'s','name':i}]
        iddi[i]=a
    df_x=df_x.replace(iddi)
    a=0
    tempEdges=[]
    for i in df_x.index:
        a+=1
        tempEdges+=[{'id':str(a),'source':str(df_x['药品'][i]),'target':str(df_x['适应症'][i]),'relations':'功能主治','value':str(1)}]
    return nodes,tempEdges
def neo4j_pipei(str1,var1,graph):
    li = graph.run("match (n:"+var1+") return n.name as n ").to_data_frame()["n"]
    scoreli,df=sdscore(str1,li)
    df["score"]=(df["cos_sim"]+df["distance"])/2
    df=df.sort_values(by="score")
    df=df.tail(1)
    return list(df["字段"])[0]
def detail_pipei(detail,graph):
    for i in detail:
        for ii in detail[i]:
            detail[i][ii]=[neo4j_pipei(m,i,graph) for m in detail[i][ii]]
    return detail
def neo4j_sim(li1,graph):
    for i in range(len(li1)):
        combinations =list(itertools.combinations(li1,i+1))
        yuju=''
        yujuli=[]
        for ii in range(i+1):
            yuju+=' match (n'+str(ii)+':`适应症`)<--(o:`药品`)-->(m:`适应症`) '
        for ii in combinations:
            yujuli+=[[]]
            for iii in range(i+1):
                yujuli[-1]+=[' n'+str(iii)+'.name contains "'+str(ii[iii])+'"']
            yujuli[-1]= '('+' and '.join(yujuli[-1])+')'
        yuju+=' where ('+' or '.join(yujuli)+')'
        for ii in range(i+1):
            for iii in range(ii):
                yuju+=" and n"+str(ii)+".name<>n"+str(iii)+".name"
        yuju+=" return m.name as m,o.name as o "
        df=graph.run(yuju).to_data_frame()
        try:
            resultli=list(pd.DataFrame(df['m'].value_counts()).sort_values(by='count',ascending=False).reset_index()["m"])
            medli_1=list(pd.DataFrame(df['o'].value_counts()).sort_values(by='count',ascending=False).reset_index()["o"])
        except:
            resultli=[]
            medli_1=[]
        if len(medli_1)<=20:
            if len(medli_1)>0:
                return resultli
                break
            else:
                if i==0:
                    return resultli
                    break
                else:
                    return resultli2
                    break
        resultli2=resultli.copy()
    return resultli
def neo4j_med(detail,graph):
    for times in range(len(detail['适应症']['需要'])):
        yuju=''
        m_node=set()
        s_node=set()
        x_node=set()
        n=0
        for i in ['适应症','剂型', '成分', '特殊人群', '用法']:
            if i=='适应症':
                combinations =list(itertools.combinations(detail[i]["需要"],times+1))
                for ii in range(times+1):
                    yuju+=" match (n:`药品`)-[x"+str(ii)+"]->(m"+str(ii)+":`"+i+"`)"
                    m_node.add('n')
                    s_node.add("m"+str(ii))
                    x_node.add("x"+str(ii))
                yujuli=[]
                for ii in combinations:
                    yujuli+=[[]]
                    for iii in range(times+1):
                        yujuli[-1]+=[" m"+str(iii)+".name contains '"+str(ii[iii])+"' "]
                    yujuli[-1]="("+" and ".join(yujuli[-1])+")"
                yuju+=" where ("+" or ".join(yujuli)+" ) "
                for ii in range(times+1):
                    for iii in range(ii):
                        yuju+=" and m"+str(ii)+".name<>m"+str(iii)+".name "
            else:
                for ii in detail[i]["不需要"]:
                    yuju+=" AND NOT EXISTS {MATCH (n)--(n"+str(n)+":`"+i+"`)WHERE n"+str(n)+".name CONTAINS '"+ii+"'} "
                    n+=1
        yuju+=(" return "+','.join(list(m_node))+','+','.join(list(s_node))+','+','.join(list(x_node)))
        df=graph.run(yuju).to_data_frame()
        if df.empty:
            medli = []
            sympli = []
            relali = []
        else:
            medli=node_di(df,m_node)
            sympli=node_di(df,s_node)
            relali=rela_di(df,x_node)
        if len(medli)<=20:
            if len(medli)>=5:
                return medli,sympli,relali
                break
            else:
                if times==0:
                    return medli,sympli,relali
                    break
                else:
                    return medli1,sympli1,relali1
                    break
        medli1=medli.copy()
        sympli1=sympli.copy()
        relali1=relali.copy()
    return medli,sympli,relali
def sort_score(series):
    score=0
    for i in detail:
        for ii in detail[i]["需要"]:
            if i=='适应症':
                iii="功效或适应症"
            elif i=="成分":
                iii="成分明细"
            if ii in series[iii]:
                if i=='适应症':
                    score+=1
                else:
                    score+=5
    return score
def med_repli(medli,detail,graph):
    for i in range(10000):
        yuju=''
        m_node=set()
        s_node=set()
        x_node=set()
        for ii in range(i+1):
            yuju+=" match (n:`药品`)-[x"+str(ii)+"x"+str(ii)+"]->(o"+str(ii)+":`适应症`)<-[x"+str(ii)+"]-(m:`药品`) where m.name in "+str(medli)+" and m.name<>n.name "
            m_node.add('n')
            s_node.add("o" + str(ii))
            x_node.add("x" + str(ii))
            x_node.add("x" + str(ii)+"x"+str(ii))
        for ii in range(i+1):
            for iii in range(ii):
                yuju+=" and o"+str(ii)+".name<>o"+str(iii)+".name "
        n=0
        for ii in detail:
            for iii in detail[ii]["不需要"]:
                yuju+=" AND NOT EXISTS {MATCH (n)--(n"+str(n)+":`"+ii+"`)WHERE n"+str(n)+".name CONTAINS '"+iii+"'} "
                n+=1
        yuju+=(" return "+','.join(list(m_node))+','+','.join(list(s_node))+','+','.join(list(x_node)))
        print(yuju)
        df=graph.run(yuju).to_data_frame()
        if df.empty:
            drugli = []
            sympli = []
            relali = []
        else:
            drugli=node_di(df,m_node)
            sympli=node_di(df,s_node)
            relali=rela_di(df,x_node)
        if len(drugli)<=20:
            if len(drugli)>5:
                return drugli,sympli,relali
                break
            else:
                if i==0:
                    return drugli,sympli,relali
                    break
                else:
                    return drugli1,sympli1,relali1
                    break
        drugli1=drugli.copy()
        sympli1=sympli.copy()
        relali1=relali.copy()
    return drugli,sympli,relali
def drug_confirm(query,mset):
    text=llm("You are a Senior doctor.","下面query中提到了medset中的哪些药品（可能只提到一个），将提到的药品名称原封不动地用集合的形式输出，例如{'依托度酸片', '来氟米特片'}。若无包含药品则输出{}，不能输出medset中不包含的药品。\n query:"+query+"\n medset:"+str(mset))
    try:
        text=re.search("\{(.*?)\}",text).group(0)
    except:
        text={}
    return ''.join(text.split()).strip("{\'").strip("\'}").split('\',\'')
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

detail={"适应症":{"需要":[],"不需要":[]},"剂型":{"需要":["软膏剂"],"不需要":[]},"用法":{"需要":["外用"],"不需要":[]},"成分":{"需要":[],"不需要":[]},"特殊人群":{"需要":[],"不需要":[]}}
# 首页
@app.route('/')
def index():
    return render_template('index.html')


# 药品查询页面
@app.route('/search')
def search():
    return render_template('search.html')


# 药品比较页面
@app.route('/compare')
def compare():
    return render_template('compare.html')


# 相似药品推荐页面
@app.route('/knowledgegraph')
def similar():
    return render_template('knowledgegraph.html')


# 基于症状的药品推荐页面
@app.route('/manual')
def smart():
    return render_template('manual.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    text = data.get('text')
    current = data.get('current_number')
    selected = data.get('selected_factor')
    type = data.get('type')
    detail=data.get('detail')
    if selected is not None and current is not None:
        if type=='b':
            sym_li = data['sym_li']
            sym_li += [selected]
            print(detail,selected)
            detail["适应症"]["需要"] = detail["适应症"]["需要"] + [selected]
            sympli = neo4j_sim(detail["适应症"]["需要"], graph)
            sympli = [x for x in sympli if x not in sym_li]
            if len(sympli)<=5:
                factors = sympli
            else:
                factors = sympli[:5]
            _med,_1,_2=neo4j_med(detail,graph)
            if len(_med)<=20:
                n_drug,_1,_2 = neo4j_med(detail, graph)
                if len(n_drug) > 20:
                    n_drug = n_drug[:20]
                massage = '\n'.join(list(meg_data[n_drug]))
                result = med_advice(current, massage)
                nodes,tempEdges=node_edge(n_drug,_1,_2)
                return jsonify({
                    'type': 'prime',
                    'result': result,
                    'medlist': list(n_drug),
                    'nodes': nodes,
                    'tempEdges': tempEdges
                })
            else:
                return jsonify({
                    'type': type,
                    'message': '请继续选择适应症或功效',
                    'detail': detail,
                    'factors': factors,
                    'current': current,
                    'sym_li':sym_li,
                    'mset':data.get('mset'),
                    'drugli':data.get('drugli'),
                })
        elif type=='c':
            sym_li=data['sym_li']
            mset=data['mset']
            drugli=data['drugli']
            sym_li += [selected]
            yuju = "match (m:`药品`)-[x]->(n:`适应症`) where m.name in " + str(drugli)
            m_node = {"m"}
            s_node = {"n"}
            x_node = {"x"}
            for i in range(len(sym_li)):
                yuju += " match (m:`药品`)-[x" + str(i) + "]->(n" + str(i) + ":`适应症`) where n" + str(i) + ".name='" + \
                        sym_li[i] + "'  "
                m_node.add('m')
                s_node.add("n" + str(i))
                x_node.add("x" + str(i))
                for ii in range(i):
                    yuju += " and n" + str(ii) + ".name<>n" + str(i) + ".name "
            yuju+=(" return "+','.join(list(m_node))+','+','.join(list(s_node))+','+','.join(list(x_node)))
            df1 = graph.run(yuju).to_data_frame()
            if df1.empty==True:
                drugli = []
                symli = []
                relali = []
            else:
                drugli = node_di(df1, m_node)
                symli = node_di(df1, s_node)
                relali = rela_di(df1, x_node)
            symli = [x for x in symli if x not in sym_li]
            if len(symli)<=5:
                factors=symli
            else:
                factors=symli[:5]
            if len(drugli)<=20:
                massage = '\n'.join(list(meg_data[drugli + list(mset)]))
                result = med_replace(current, massage)
                nodes, tempEdges = node_edge(drugli + list(mset), symli,relali)
                return jsonify({
                    'type': 'prime',
                    'result': result,
                    'medlist': list(drugli + list(mset)),
                    'nodes': nodes,
                    'tempEdges': tempEdges
                })
            else:
                return jsonify({
                    'type': type,
                    'message': '请继续选择适应症或功效',
                    'factors': factors,
                    'detail': detail,
                    'current': current,
                    'sym_li': sym_li,
                    'mset': list(mset),
                    'drugli': drugli,
                })
    else:
        query=text
        query = llm("You are a drug seeker with a strong medical background.",
                    '请将以下句子用更专业的语言表述(仅输出改写后的句子，并用《》括起来，请按照规定格式输出，不要添加其他内容):\n' + query)
        query = re.search("《(.*?)》", query).group(1)
        query_c = query_type(query)
        query_c = re.findall("《(.*?)》", query_c)
        query_c=query_c[0]
        print(query_c)
        if query_c=='a':
            medli = eval(query_med_get(query))
            df = pipei(list(meg_data.index), medli)
            mset = set()
            for i in df:
                if len(df[df[i] == df[i].max()].index) == 1:
                    mset = mset | {df[df[i] == df[i].max()].index[0]}
            if len(medli) != 0:
                query = query_med_repl(query, mset)
                query = re.search("《(.*?)》", query).group(1)
                mset1=drug_confirm(query,mset)
                if len(mset1)==0:
                    mset1=mset
                mset=mset1
                massage = '\n'.join(list(meg_data[list(mset)]))
            else:
                massage=""
                mset=[]
            result = med_search(query, massage)
            df = graph.run(
                "match (m:`药品`)-[x]->(s:`适应症`) where m.name in "+str(list(mset))+" return m,x,s").to_data_frame()
            nodes, tempEdges = node_edge(node_di(df, {'m'}), node_di(df, {'s'}), rela_di(df, {'x'}))
            return jsonify({
                'type': query_c,
                'result': result,
                'medlist': list(mset),
                'nodes': nodes,
                'tempEdges': tempEdges
            })
        elif query_c=='b':
            detail = eval(query_analy(query))
            detail = detail_pipei(detail, graph)
            if len(detail["适应症"]["需要"])>0:
                sympli = neo4j_sim(detail["适应症"]["需要"], graph)
                sym_li = []
                drugli=[]
                mset = set()
                if len(sympli) > 5:
                    sympli = sympli[:5]
                factors=list(sympli)
                _med,_1,_2=neo4j_med(detail, graph)
                print(_med)
                anna=0
            else:
                anna=1
            if len(_med) <= 20 and anna==0:
                n_drug,_1,_2 = neo4j_med(detail, graph)
                if len(n_drug) > 20:
                    n_drug = n_drug[:20]
                massage = '\n'.join(list(meg_data[n_drug]))
                result = med_advice(query, massage)
                nodes, tempEdges = node_edge(n_drug, _1, _2)
                return jsonify({
                    'type': 'prime',
                    'result': result,
                    'medlist': list(n_drug),
                    'nodes': nodes,
                    'tempEdges': tempEdges
                })
            elif anna==1:
                result = med_advice(query, "")
                return jsonify({
                    'type': 'prime',
                    'result': result,
                    'medlist': [],
                    'nodes': [],
                    'tempEdges': []
                })
            else:
                return jsonify({
                    'type': query_c,
                    'message': '请选择您希望推荐的药品有什么适应症或功效',
                    'factors': factors,
                    'original': query,
                    'detail': detail,
                    'sym_li': sym_li,
                    'mset': list(mset),
                    'drugli': drugli,
                })

        elif query_c=='c':
            medli = eval(query_med_get(query))
            df = pipei(list(meg_data.index), medli)
            mset = set()
            for i in df:
                if len(df[df[i] == df[i].max()].index) == 1:
                    mset = mset | {df[df[i] == df[i].max()].index[0]}
            if len(medli) != 0:
                query = query_med_repl(query, mset)
                query = re.search("《(.*?)》", query).group(1)
                mset = drug_confirm(query, mset)
                detail = eval(query_analy_sub(query))
                sym_li = []
                drugli,_1,_2= med_repli(list(mset), detail, graph)
                if len(drugli)>0:
                    df1 = graph.run("match (m:`药品`)-->(n:`适应症`) where m.name in " + str(
                        drugli) + " return m.name as m,n.name as n").to_data_frame()
                    symli = pd.DataFrame(df1['n'].value_counts()).reset_index().sort_values(by="n", ascending=False)["n"]
                    symli = [x for x in symli if x not in sym_li]
                    if len(symli) > 5:
                        symli = symli[:5]
                    factors = list(symli)
                    anna=0
                else:
                    anna=1
            else:
                anna=1
            if len(drugli)<=20 and anna==0:
                massage = '\n'.join(list(meg_data[drugli + list(mset)]))
                result = med_replace(query, massage)
                nodes, tempEdges = node_edge(drugli+list(mset), _1, _2)
                return jsonify({
                    'type': 'prime',
                    'result': result,
                    'medlist': list(drugli + list(mset)),
                    'nodes': nodes,
                    'tempEdges': tempEdges
                })
            elif anna==1:
                result = med_replace(query, '\n'.join(list(meg_data[list(mset)])))
                return jsonify({
                    'type': 'prime',
                    'result': result,
                    'medlist': [],
                    'nodes': [],
                    'tempEdges':[]
                })
            else:
                return jsonify({
                    'type': query_c,
                    'message': "请选择您希望替代的药品有什么适应症或功效",
                    'factors': factors,
                    'original': query,
                    'detail': detail,
                    'sym_li': sym_li,
                    'mset': list(mset),
                    'drugli': drugli,
                })
        elif query_c=='d':
            medli = eval(query_med_get(query))
            df = pipei(list(meg_data.index), medli)
            mset = set()
            for i in df:
                if len(df[df[i] == df[i].max()].index) == 1:
                    mset = mset | {df[df[i] == df[i].max()].index[0]}
            if len(medli) != 0:
                query = query_med_repl(query, mset)
                query = re.search("《(.*?)》", query).group(1)
                mset = drug_confirm(query, mset)
                massage = '\n'.join(list(meg_data[list(mset)]))
            else:
                massage = ''
            result = med_compare(query, massage)
            df = graph.run(
                "match (m:`药品`)-[x]->(s:`适应症`) where m.name in " + str(
                    list(mset)) + " return m,x,s").to_data_frame()
            nodes, tempEdges = node_edge(node_di(df, {'m'}), node_di(df, {'s'}), rela_di(df, {'x'}))
            return jsonify({
                'type': query_c,
                'result': result,
                'medlist': list(mset),
                'nodes': nodes,
                'tempEdges': tempEdges
            })

@app.route('/med')
def med():
    medname = request.args.get('medname')
    kwargs = {
        'tymc': getvar(meg_data_orig,"通用名称",medname),
        'spmc': getvar(meg_data_orig,"商品名称",medname),
        'ywmc': getvar(meg_data_orig,"英文名称",medname),
        'hypy': getvar(meg_data_orig,"汉语拼音",medname),
        'zdj': getvar(meg_data_orig, "指导价", medname),
        'cfy': getvar(meg_data_orig, "处方药", medname),
        'cf': getvar(meg_data_orig, "成分", medname),
        'xz': getvar(meg_data_orig, "性状", medname),
        'syz': getvar(meg_data_orig, "适应症", medname),
        'yfyl': getvar(meg_data_orig, "用法用量", medname),
        'blfy': getvar(meg_data_orig, "不良反应", medname),
        'jj': getvar(meg_data_orig, "禁忌", medname),
        'zysx': getvar(meg_data_orig, "注意事项", medname),
        'tsrqyy': getvar(meg_data_orig, "特殊人群用药", medname),
        'ywxhzy': getvar(meg_data_orig, "药物相互作用", medname),
        'ylzy': getvar(meg_data_orig, "药理作用", medname),
        'zc': getvar(meg_data_orig, "贮藏", medname),
        'gg': getvar(meg_data_orig, "规格", medname),
        'bzgg': getvar(meg_data_orig, "包装规格", medname),
        'yxq': getvar(meg_data_orig, "有效期", medname),
    }
    return render_template("med.html", **kwargs)

# API接口 - 搜索药品
@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json()
    query  = data.get('text')
    if not query:
        return jsonify({'error': '请输入搜索关键词'}), 400

    medli = eval(query_med_get(query))
    df = pipei(list(meg_data.index), medli)
    mset = set()
    for i in df:
        if len(df[df[i] == df[i].max()].index) == 1:
            mset = mset | {df[df[i] == df[i].max()].index[0]}
    if len(medli) != 0:
        query = query_med_repl(query, mset)
        query = re.search("《(.*?)》", query).group(1)
        mset1 = drug_confirm(query, mset)
        if len(mset1) == 0:
            mset1 = mset
        mset = mset1
    else:
        mset = []
    mdetail=[]
    print(mset)
    for i in list(mset):
        mdetail+=[{"药品名称":getvar(meg_data_orig,"通用名称",i),
                   "适应症":getvar(meg_data_orig,"适应症",i),
                   "成分": list(eval(getvar(pd.read_csv("static/data/药品通数据_知识图谱.csv",encoding="ANSI"), "成分明细", i))),
                   "用法用量":getvar(meg_data_orig,"用法用量",i),
                   "禁忌":getvar(meg_data_orig,"禁忌",i),
                   "价格":getvar(meg_data_orig,"指导价",i)}]
    return jsonify(mdetail)


# API接口 - 药品比较
@app.route('/api/compare', methods=['POST'])
def api_compare():
    data = request.get_json()
    query  = '；'.join(data.get('drugs'))
    if not query:
        return jsonify({'error': '请输入搜索关键词'}), 400

    medli = data.get('drugs')
    print(medli)
    df = pipei(list(meg_data.index), medli)
    mset = set()
    for i in df:
        if len(df[df[i] == df[i].max()].index) == 1:
            mset = mset | {df[df[i] == df[i].max()].index[0]}
    mdetail=[]
    print(mset)
    for m in list(mset):
        mdetail+=[{
        'tymc': getvar(meg_data_orig,"通用名称",m),
        'spmc': getvar(meg_data_orig,"商品名称",m),
        'ywmc': getvar(meg_data_orig,"英文名称",m),
        'hypy': getvar(meg_data_orig,"汉语拼音",m),
        'zdj': getvar(meg_data_orig, "指导价", m),
        'cfy': getvar(meg_data_orig, "处方药", m),
        'cf': getvar(meg_data_orig, "成分", m),
        'xz': getvar(meg_data_orig, "性状", m),
        'syz': getvar(meg_data_orig, "适应症", m),
        'yfyl': getvar(meg_data_orig, "用法用量", m),
        'blfy': getvar(meg_data_orig, "不良反应", m),
        'jj': getvar(meg_data_orig, "禁忌", m),
        'zysx': getvar(meg_data_orig, "注意事项", m),
        'tsrqyy': getvar(meg_data_orig, "特殊人群用药", m),
        'ywxhzy': getvar(meg_data_orig, "药物相互作用", m),
        'ylzy': getvar(meg_data_orig, "药理作用", m),
        'zc': getvar(meg_data_orig, "贮藏", m),
        'gg': getvar(meg_data_orig, "规格", m),
        'bzgg': getvar(meg_data_orig, "包装规格", m),
        'yxq': getvar(meg_data_orig, "有效期", m)
    }]
    return jsonify(mdetail)

# API接口 - 相似药品推荐
@app.route('/api/node', methods=['POST'])
def api_node():
    data = request.get_json()
    df=graph.run("MATCH (a:"+data['type']+")-[x]-(b) WHERE a.name = '"+data['keyword']+"' RETURN a,b,x").to_data_frame()
    if df.empty:
        medli = []
        sympli = []
        relali = []
    else:
        medli = node_di(df,{'a'})
        sympli = node_di(df,{'b'})
        relali = rela_di(df,{'x'})
    nodes, tempEdges = node_edge(medli, sympli , relali)
    return jsonify({'nodes': nodes, 'edges': tempEdges})

@app.route('/api/relation', methods=['POST'])
def api_relation():
    data = request.get_json()
    if data['start']=="":
        df=graph.run("MATCH (a)-[x:"+data['rel']+"]->(b) WHERE b.name='"+data['end']+"' RETURN a,b,x").to_data_frame()
    elif data['end']=="":
        df=graph.run("MATCH (a)-[x:"+data['rel']+"]->(b) WHERE a.name = '"+data["start"]+"' RETURN a,b,x").to_data_frame()
    else:
        df=graph.run("MATCH (a)-[x:"+data['rel']+"]->(b) WHERE a.name = '"+data["start"]+"' and b.name='"+data['end']+"' RETURN a,b,x").to_data_frame()
    if df.empty:
        medli = []
        sympli = []
        relali = []
    else:
        medli = node_di(df,{'a'})
        sympli = node_di(df,{'b'})
        relali = rela_di(df,{'x'})
    nodes, tempEdges = node_edge(medli, sympli , relali)
    return jsonify({'nodes': nodes, 'edges': tempEdges})


if __name__ == '__main__':
    app.run(debug=True)