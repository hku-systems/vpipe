import sys
sys.path.append("..")
from vpipe import Stage
from vpipe import Bert

def model(criterion, partition, recompute_ratio):
    _declares = get_declares()
    _calculations = get_caculations()
    module = Bert(_declares, _calculations)
    module.generate_layer_blocks()
    start = 0
    inputs = []
    outputs = [['out490']]
    all_outputs = []
    declares = []
    calculations = []
    for i in partition:
        stage = module.generate_stage(start, start + i)
        start += i
        declares.append(stage[0])
        calculations.append(stage[1])
        inputs.append(stage[2])
        all_outputs.append(stage[3])
    
    for i in range(len(partition)-1, 0, -1):
        previous_output = []
        for name in inputs[i]:
            if name != 'out0' and name != 'out1' and name != 'out2':
                previous_output.append(name)
        for name in outputs[0]:
            if name not in all_outputs[i] and name not in previous_output:
                previous_output.append(name)
        outputs.insert(0, previous_output)

    return [
        (Stage(inputs[0], outputs[0], declares[0], calculations[0], recompute_ratio[0]), replace(inputs[0]), outputs[0]),
        (Stage(inputs[1], outputs[1], declares[1], calculations[1], recompute_ratio[1]), replace(inputs[1]), outputs[1]),
        (Stage(inputs[2], outputs[2], declares[2], calculations[2], recompute_ratio[2]), replace(inputs[2]), outputs[2]),
        (Stage(inputs[3], outputs[3], declares[3], calculations[3], recompute_ratio[3]), replace(inputs[3]), outputs[3]),
        (Stage(inputs[4], outputs[4], declares[4], calculations[4], recompute_ratio[4]), replace(inputs[4]), outputs[4]),
        (Stage(inputs[5], outputs[5], declares[5], calculations[5], recompute_ratio[5]), replace(inputs[5]), outputs[5]),
        (Stage(inputs[6], outputs[6], declares[6], calculations[6], recompute_ratio[6]), replace(inputs[6]), outputs[6]),
        (Stage(inputs[7], outputs[7], declares[7], calculations[7], recompute_ratio[7]), replace(inputs[7]), outputs[7]),
        (criterion, outputs[7], ["loss"])
    ]

def replace(inputs):
    for i in range(len(inputs)):
        if inputs[i] == 'out0':
            inputs[i] = 'input0'
        elif inputs[i] == 'out1':
            inputs[i] = 'input1'
        elif inputs[i] == 'out2':
            inputs[i] = 'input2'
    return inputs

def get_declares():
    return '''self.layer6 = BertEmbeddings(30528, 1024, 512, 2, 0.1)
self.layer7 = BertSelfAttention(1024, 16, 0.1)
self.layer8 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer9 = torch.nn.Dropout(p=0.1)
self.layer11 = BertLayerNorm(1024)
self.layer12 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer13 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer14 = torch.nn.Dropout(p=0.1)
self.layer16 = BertLayerNorm(1024)
self.layer17 = BertSelfAttention(1024, 16, 0.1)
self.layer18 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer19 = torch.nn.Dropout(p=0.1)
self.layer21 = BertLayerNorm(1024)
self.layer22 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer23 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer24 = torch.nn.Dropout(p=0.1)
self.layer26 = BertLayerNorm(1024)
self.layer27 = BertSelfAttention(1024, 16, 0.1)
self.layer28 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer29 = torch.nn.Dropout(p=0.1)
self.layer31 = BertLayerNorm(1024)
self.layer32 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer33 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer34 = torch.nn.Dropout(p=0.1)
self.layer36 = BertLayerNorm(1024)
self.layer37 = BertSelfAttention(1024, 16, 0.1)
self.layer38 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer39 = torch.nn.Dropout(p=0.1)
self.layer41 = BertLayerNorm(1024)
self.layer42 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer43 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer44 = torch.nn.Dropout(p=0.1)
self.layer46 = BertLayerNorm(1024)
self.layer47 = BertSelfAttention(1024, 16, 0.1)
self.layer48 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer49 = torch.nn.Dropout(p=0.1)
self.layer51 = BertLayerNorm(1024)
self.layer52 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer53 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer54 = torch.nn.Dropout(p=0.1)
self.layer56 = BertLayerNorm(1024)
self.layer57 = BertSelfAttention(1024, 16, 0.1)
self.layer58 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer59 = torch.nn.Dropout(p=0.1)
self.layer61 = BertLayerNorm(1024)
self.layer62 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer63 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer64 = torch.nn.Dropout(p=0.1)
self.layer66 = BertLayerNorm(1024)
self.layer67 = BertSelfAttention(1024, 16, 0.1)
self.layer68 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer69 = torch.nn.Dropout(p=0.1)
self.layer71 = BertLayerNorm(1024)
self.layer72 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer73 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer74 = torch.nn.Dropout(p=0.1)
self.layer76 = BertLayerNorm(1024)
self.layer77 = BertSelfAttention(1024, 16, 0.1)
self.layer78 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer79 = torch.nn.Dropout(p=0.1)
self.layer81 = BertLayerNorm(1024)
self.layer82 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer83 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer84 = torch.nn.Dropout(p=0.1)
self.layer86 = BertLayerNorm(1024)
self.layer87 = BertSelfAttention(1024, 16, 0.1)
self.layer88 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer89 = torch.nn.Dropout(p=0.1)
self.layer91 = BertLayerNorm(1024)
self.layer92 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer93 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer94 = torch.nn.Dropout(p=0.1)
self.layer96 = BertLayerNorm(1024)
self.layer97 = BertSelfAttention(1024, 16, 0.1)
self.layer98 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer99 = torch.nn.Dropout(p=0.1)
self.layer101 = BertLayerNorm(1024)
self.layer102 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer103 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer104 = torch.nn.Dropout(p=0.1)
self.layer106 = BertLayerNorm(1024)
self.layer107 = BertSelfAttention(1024, 16, 0.1)
self.layer108 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer109 = torch.nn.Dropout(p=0.1)
self.layer111 = BertLayerNorm(1024)
self.layer112 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer113 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer114 = torch.nn.Dropout(p=0.1)
self.layer116 = BertLayerNorm(1024)
self.layer117 = BertSelfAttention(1024, 16, 0.1)
self.layer118 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer119 = torch.nn.Dropout(p=0.1)
self.layer121 = BertLayerNorm(1024)
self.layer122 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer123 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer124 = torch.nn.Dropout(p=0.1)
self.layer126 = BertLayerNorm(1024)
self.layer127 = BertSelfAttention(1024, 16, 0.1)
self.layer128 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer129 = torch.nn.Dropout(p=0.1)
self.layer131 = BertLayerNorm(1024)
self.layer132 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer133 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer134 = torch.nn.Dropout(p=0.1)
self.layer136 = BertLayerNorm(1024)
self.layer137 = BertSelfAttention(1024, 16, 0.1)
self.layer138 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer139 = torch.nn.Dropout(p=0.1)
self.layer141 = BertLayerNorm(1024)
self.layer142 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer143 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer144 = torch.nn.Dropout(p=0.1)
self.layer146 = BertLayerNorm(1024)
self.layer147 = BertSelfAttention(1024, 16, 0.1)
self.layer148 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer149 = torch.nn.Dropout(p=0.1)
self.layer151 = BertLayerNorm(1024)
self.layer152 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer153 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer154 = torch.nn.Dropout(p=0.1)
self.layer156 = BertLayerNorm(1024)
self.layer157 = BertSelfAttention(1024, 16, 0.1)
self.layer158 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer159 = torch.nn.Dropout(p=0.1)
self.layer161 = BertLayerNorm(1024)
self.layer162 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer163 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer164 = torch.nn.Dropout(p=0.1)
self.layer166 = BertLayerNorm(1024)
self.layer167 = BertSelfAttention(1024, 16, 0.1)
self.layer168 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer169 = torch.nn.Dropout(p=0.1)
self.layer171 = BertLayerNorm(1024)
self.layer172 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer173 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer174 = torch.nn.Dropout(p=0.1)
self.layer176 = BertLayerNorm(1024)
self.layer177 = BertSelfAttention(1024, 16, 0.1)
self.layer178 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer179 = torch.nn.Dropout(p=0.1)
self.layer181 = BertLayerNorm(1024)
self.layer182 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer183 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer184 = torch.nn.Dropout(p=0.1)
self.layer186 = BertLayerNorm(1024)
self.layer187 = BertSelfAttention(1024, 16, 0.1)
self.layer188 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer189 = torch.nn.Dropout(p=0.1)
self.layer191 = BertLayerNorm(1024)
self.layer192 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer193 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer194 = torch.nn.Dropout(p=0.1)
self.layer196 = BertLayerNorm(1024)
self.layer197 = BertSelfAttention(1024, 16, 0.1)
self.layer198 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer199 = torch.nn.Dropout(p=0.1)
self.layer201 = BertLayerNorm(1024)
self.layer202 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer203 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer204 = torch.nn.Dropout(p=0.1)
self.layer206 = BertLayerNorm(1024)
self.layer207 = BertSelfAttention(1024, 16, 0.1)
self.layer208 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer209 = torch.nn.Dropout(p=0.1)
self.layer211 = BertLayerNorm(1024)
self.layer212 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer213 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer214 = torch.nn.Dropout(p=0.1)
self.layer216 = BertLayerNorm(1024)
self.layer217 = BertSelfAttention(1024, 16, 0.1)
self.layer218 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer219 = torch.nn.Dropout(p=0.1)
self.layer221 = BertLayerNorm(1024)
self.layer222 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer223 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer224 = torch.nn.Dropout(p=0.1)
self.layer226 = BertLayerNorm(1024)
self.layer227 = BertSelfAttention(1024, 16, 0.1)
self.layer228 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer229 = torch.nn.Dropout(p=0.1)
self.layer231 = BertLayerNorm(1024)
self.layer232 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer233 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer234 = torch.nn.Dropout(p=0.1)
self.layer236 = BertLayerNorm(1024)
self.layer237 = BertSelfAttention(1024, 16, 0.1)
self.layer238 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer239 = torch.nn.Dropout(p=0.1)
self.layer241 = BertLayerNorm(1024)
self.layer242 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer243 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer244 = torch.nn.Dropout(p=0.1)
self.layer246 = BertLayerNorm(1024)
self.layer247 = BertSelfAttention(1024, 16, 0.1)
self.layer248 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer249 = torch.nn.Dropout(p=0.1)
self.layer251 = BertLayerNorm(1024)
self.layer252 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer253 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer254 = torch.nn.Dropout(p=0.1)
self.layer256 = BertLayerNorm(1024)
self.layer257 = BertSelfAttention(1024, 16, 0.1)
self.layer258 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer259 = torch.nn.Dropout(p=0.1)
self.layer261 = BertLayerNorm(1024)
self.layer262 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer263 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer264 = torch.nn.Dropout(p=0.1)
self.layer266 = BertLayerNorm(1024)
self.layer267 = BertSelfAttention(1024, 16, 0.1)
self.layer268 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer269 = torch.nn.Dropout(p=0.1)
self.layer271 = BertLayerNorm(1024)
self.layer272 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer273 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer274 = torch.nn.Dropout(p=0.1)
self.layer276 = BertLayerNorm(1024)
self.layer277 = BertSelfAttention(1024, 16, 0.1)
self.layer278 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer279 = torch.nn.Dropout(p=0.1)
self.layer281 = BertLayerNorm(1024)
self.layer282 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer283 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer284 = torch.nn.Dropout(p=0.1)
self.layer286 = BertLayerNorm(1024)
self.layer287 = BertSelfAttention(1024, 16, 0.1)
self.layer288 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer289 = torch.nn.Dropout(p=0.1)
self.layer291 = BertLayerNorm(1024)
self.layer292 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer293 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer294 = torch.nn.Dropout(p=0.1)
self.layer296 = BertLayerNorm(1024)
self.layer297 = BertSelfAttention(1024, 16, 0.1)
self.layer298 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer299 = torch.nn.Dropout(p=0.1)
self.layer301 = BertLayerNorm(1024)
self.layer302 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer303 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer304 = torch.nn.Dropout(p=0.1)
self.layer306 = BertLayerNorm(1024)
self.layer307 = BertSelfAttention(1024, 16, 0.1)
self.layer308 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer309 = torch.nn.Dropout(p=0.1)
self.layer311 = BertLayerNorm(1024)
self.layer312 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer313 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer314 = torch.nn.Dropout(p=0.1)
self.layer316 = BertLayerNorm(1024)
self.layer317 = BertSelfAttention(1024, 16, 0.1)
self.layer318 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer319 = torch.nn.Dropout(p=0.1)
self.layer321 = BertLayerNorm(1024)
self.layer322 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer323 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer324 = torch.nn.Dropout(p=0.1)
self.layer326 = BertLayerNorm(1024)
self.layer327 = BertSelfAttention(1024, 16, 0.1)
self.layer328 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer329 = torch.nn.Dropout(p=0.1)
self.layer331 = BertLayerNorm(1024)
self.layer332 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer333 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer334 = torch.nn.Dropout(p=0.1)
self.layer336 = BertLayerNorm(1024)
self.layer337 = BertSelfAttention(1024, 16, 0.1)
self.layer338 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer339 = torch.nn.Dropout(p=0.1)
self.layer341 = BertLayerNorm(1024)
self.layer342 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer343 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer344 = torch.nn.Dropout(p=0.1)
self.layer346 = BertLayerNorm(1024)
self.layer347 = BertSelfAttention(1024, 16, 0.1)
self.layer348 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer349 = torch.nn.Dropout(p=0.1)
self.layer351 = BertLayerNorm(1024)
self.layer352 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer353 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer354 = torch.nn.Dropout(p=0.1)
self.layer356 = BertLayerNorm(1024)
self.layer357 = BertSelfAttention(1024, 16, 0.1)
self.layer358 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer359 = torch.nn.Dropout(p=0.1)
self.layer361 = BertLayerNorm(1024)
self.layer362 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer363 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer364 = torch.nn.Dropout(p=0.1)
self.layer366 = BertLayerNorm(1024)
self.layer367 = BertSelfAttention(1024, 16, 0.1)
self.layer368 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer369 = torch.nn.Dropout(p=0.1)
self.layer371 = BertLayerNorm(1024)
self.layer372 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer373 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer374 = torch.nn.Dropout(p=0.1)
self.layer376 = BertLayerNorm(1024)
self.layer377 = BertSelfAttention(1024, 16, 0.1)
self.layer378 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer379 = torch.nn.Dropout(p=0.1)
self.layer381 = BertLayerNorm(1024)
self.layer382 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer383 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer384 = torch.nn.Dropout(p=0.1)
self.layer386 = BertLayerNorm(1024)
self.layer387 = BertSelfAttention(1024, 16, 0.1)
self.layer388 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer389 = torch.nn.Dropout(p=0.1)
self.layer391 = BertLayerNorm(1024)
self.layer392 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer393 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer394 = torch.nn.Dropout(p=0.1)
self.layer396 = BertLayerNorm(1024)
self.layer397 = BertSelfAttention(1024, 16, 0.1)
self.layer398 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer399 = torch.nn.Dropout(p=0.1)
self.layer401 = BertLayerNorm(1024)
self.layer402 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer403 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer404 = torch.nn.Dropout(p=0.1)
self.layer406 = BertLayerNorm(1024)
self.layer407 = BertSelfAttention(1024, 16, 0.1)
self.layer408 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer409 = torch.nn.Dropout(p=0.1)
self.layer411 = BertLayerNorm(1024)
self.layer412 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer413 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer414 = torch.nn.Dropout(p=0.1)
self.layer416 = BertLayerNorm(1024)
self.layer417 = BertSelfAttention(1024, 16, 0.1)
self.layer418 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer419 = torch.nn.Dropout(p=0.1)
self.layer421 = BertLayerNorm(1024)
self.layer422 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer423 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer424 = torch.nn.Dropout(p=0.1)
self.layer426 = BertLayerNorm(1024)
self.layer427 = BertSelfAttention(1024, 16, 0.1)
self.layer428 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer429 = torch.nn.Dropout(p=0.1)
self.layer431 = BertLayerNorm(1024)
self.layer432 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer433 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer434 = torch.nn.Dropout(p=0.1)
self.layer436 = BertLayerNorm(1024)
self.layer437 = BertSelfAttention(1024, 16, 0.1)
self.layer438 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer439 = torch.nn.Dropout(p=0.1)
self.layer441 = BertLayerNorm(1024)
self.layer442 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer443 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer444 = torch.nn.Dropout(p=0.1)
self.layer446 = BertLayerNorm(1024)
self.layer447 = BertSelfAttention(1024, 16, 0.1)
self.layer448 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer449 = torch.nn.Dropout(p=0.1)
self.layer451 = BertLayerNorm(1024)
self.layer452 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer453 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer454 = torch.nn.Dropout(p=0.1)
self.layer456 = BertLayerNorm(1024)
self.layer457 = BertSelfAttention(1024, 16, 0.1)
self.layer458 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer459 = torch.nn.Dropout(p=0.1)
self.layer461 = BertLayerNorm(1024)
self.layer462 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer463 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer464 = torch.nn.Dropout(p=0.1)
self.layer466 = BertLayerNorm(1024)
self.layer467 = BertSelfAttention(1024, 16, 0.1)
self.layer468 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer469 = torch.nn.Dropout(p=0.1)
self.layer471 = BertLayerNorm(1024)
self.layer472 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer473 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer474 = torch.nn.Dropout(p=0.1)
self.layer476 = BertLayerNorm(1024)
self.layer477 = BertSelfAttention(1024, 16, 0.1)
self.layer478 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
self.layer479 = torch.nn.Dropout(p=0.1)
self.layer481 = BertLayerNorm(1024)
self.layer482 = LinearActivation(in_features=1024, out_features=4096, bias=True)
self.layer483 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer484 = torch.nn.Dropout(p=0.1)
self.layer486 = BertLayerNorm(1024)
self.layer487 = LinearActivation(in_features=1024, out_features=1024, bias=True)
self.layer488 = BertLayerNorm(1024)
self.layer489 = torch.nn.Linear(in_features=1024, out_features=30528, bias=False)
self.layer490 = BertAdd(30528)'''

def get_caculations():
    return '''out6 = self.layer6(out0, out1)
out7 = self.layer7(out6, out2)
out8 = self.layer8(out7)
out9 = self.layer9(out8)
out9 = out9 + out6
out11 = self.layer11(out9)
out12 = self.layer12(out11)
out13 = self.layer13(out12)
out14 = self.layer14(out13)
out14 = out14 + out11
out16 = self.layer16(out14)
out17 = self.layer17(out16, out2)
out18 = self.layer18(out17)
out19 = self.layer19(out18)
out19 = out19 + out16
out21 = self.layer21(out19)
out22 = self.layer22(out21)
out23 = self.layer23(out22)
out24 = self.layer24(out23)
out24 = out24 + out21
out26 = self.layer26(out24)
out27 = self.layer27(out26, out2)
out28 = self.layer28(out27)
out29 = self.layer29(out28)
out29 = out29 + out26
out31 = self.layer31(out29)
out32 = self.layer32(out31)
out33 = self.layer33(out32)
out34 = self.layer34(out33)
out34 = out34 + out31
out36 = self.layer36(out34)
out37 = self.layer37(out36, out2)
out38 = self.layer38(out37)
out39 = self.layer39(out38)
out39 = out39 + out36
out41 = self.layer41(out39)
out42 = self.layer42(out41)
out43 = self.layer43(out42)
out44 = self.layer44(out43)
out44 = out44 + out41
out46 = self.layer46(out44)
out47 = self.layer47(out46, out2)
out48 = self.layer48(out47)
out49 = self.layer49(out48)
out49 = out49 + out46
out51 = self.layer51(out49)
out52 = self.layer52(out51)
out53 = self.layer53(out52)
out54 = self.layer54(out53)
out54 = out54 + out51
out56 = self.layer56(out54)
out57 = self.layer57(out56, out2)
out58 = self.layer58(out57)
out59 = self.layer59(out58)
out59 = out59 + out56
out61 = self.layer61(out59)
out62 = self.layer62(out61)
out63 = self.layer63(out62)
out64 = self.layer64(out63)
out64 = out64 + out61
out66 = self.layer66(out64)
out67 = self.layer67(out66, out2)
out68 = self.layer68(out67)
out69 = self.layer69(out68)
out69 = out69 + out66
out71 = self.layer71(out69)
out72 = self.layer72(out71)
out73 = self.layer73(out72)
out74 = self.layer74(out73)
out74 = out74 + out71
out76 = self.layer76(out74)
out77 = self.layer77(out76, out2)
out78 = self.layer78(out77)
out79 = self.layer79(out78)
out79 = out79 + out76
out81 = self.layer81(out79)
out82 = self.layer82(out81)
out83 = self.layer83(out82)
out84 = self.layer84(out83)
out84 = out84 + out81
out86 = self.layer86(out84)
out87 = self.layer87(out86, out2)
out88 = self.layer88(out87)
out89 = self.layer89(out88)
out89 = out89 + out86
out91 = self.layer91(out89)
out92 = self.layer92(out91)
out93 = self.layer93(out92)
out94 = self.layer94(out93)
out94 = out94 + out91
out96 = self.layer96(out94)
out97 = self.layer97(out96, out2)
out98 = self.layer98(out97)
out99 = self.layer99(out98)
out99 = out99 + out96
out101 = self.layer101(out99)
out102 = self.layer102(out101)
out103 = self.layer103(out102)
out104 = self.layer104(out103)
out104 = out104 + out101
out106 = self.layer106(out104)
out107 = self.layer107(out106, out2)
out108 = self.layer108(out107)
out109 = self.layer109(out108)
out109 = out109 + out106
out111 = self.layer111(out109)
out112 = self.layer112(out111)
out113 = self.layer113(out112)
out114 = self.layer114(out113)
out114 = out114 + out111
out116 = self.layer116(out114)
out117 = self.layer117(out116, out2)
out118 = self.layer118(out117)
out119 = self.layer119(out118)
out119 = out119 + out116
out121 = self.layer121(out119)
out122 = self.layer122(out121)
out123 = self.layer123(out122)
out124 = self.layer124(out123)
out124 = out124 + out121
out126 = self.layer126(out124)
out127 = self.layer127(out126, out2)
out128 = self.layer128(out127)
out129 = self.layer129(out128)
out129 = out129 + out126
out131 = self.layer131(out129)
out132 = self.layer132(out131)
out133 = self.layer133(out132)
out134 = self.layer134(out133)
out134 = out134 + out131
out136 = self.layer136(out134)
out137 = self.layer137(out136, out2)
out138 = self.layer138(out137)
out139 = self.layer139(out138)
out139 = out139 + out136
out141 = self.layer141(out139)
out142 = self.layer142(out141)
out143 = self.layer143(out142)
out144 = self.layer144(out143)
out144 = out144 + out141
out146 = self.layer146(out144)
out147 = self.layer147(out146, out2)
out148 = self.layer148(out147)
out149 = self.layer149(out148)
out149 = out149 + out146
out151 = self.layer151(out149)
out152 = self.layer152(out151)
out153 = self.layer153(out152)
out154 = self.layer154(out153)
out154 = out154 + out151
out156 = self.layer156(out154)
out157 = self.layer157(out156, out2)
out158 = self.layer158(out157)
out159 = self.layer159(out158)
out159 = out159 + out156
out161 = self.layer161(out159)
out162 = self.layer162(out161)
out163 = self.layer163(out162)
out164 = self.layer164(out163)
out164 = out164 + out161
out166 = self.layer166(out164)
out167 = self.layer167(out166, out2)
out168 = self.layer168(out167)
out169 = self.layer169(out168)
out169 = out169 + out166
out171 = self.layer171(out169)
out172 = self.layer172(out171)
out173 = self.layer173(out172)
out174 = self.layer174(out173)
out174 = out174 + out171
out176 = self.layer176(out174)
out177 = self.layer177(out176, out2)
out178 = self.layer178(out177)
out179 = self.layer179(out178)
out179 = out179 + out176
out181 = self.layer181(out179)
out182 = self.layer182(out181)
out183 = self.layer183(out182)
out184 = self.layer184(out183)
out184 = out184 + out181
out186 = self.layer186(out184)
out187 = self.layer187(out186, out2)
out188 = self.layer188(out187)
out189 = self.layer189(out188)
out189 = out189 + out186
out191 = self.layer191(out189)
out192 = self.layer192(out191)
out193 = self.layer193(out192)
out194 = self.layer194(out193)
out194 = out194 + out191
out196 = self.layer196(out194)
out197 = self.layer197(out196, out2)
out198 = self.layer198(out197)
out199 = self.layer199(out198)
out199 = out199 + out196
out201 = self.layer201(out199)
out202 = self.layer202(out201)
out203 = self.layer203(out202)
out204 = self.layer204(out203)
out204 = out204 + out201
out206 = self.layer206(out204)
out207 = self.layer207(out206, out2)
out208 = self.layer208(out207)
out209 = self.layer209(out208)
out209 = out209 + out206
out211 = self.layer211(out209)
out212 = self.layer212(out211)
out213 = self.layer213(out212)
out214 = self.layer214(out213)
out214 = out214 + out211
out216 = self.layer216(out214)
out217 = self.layer217(out216, out2)
out218 = self.layer218(out217)
out219 = self.layer219(out218)
out219 = out219 + out216
out221 = self.layer221(out219)
out222 = self.layer222(out221)
out223 = self.layer223(out222)
out224 = self.layer224(out223)
out224 = out224 + out221
out226 = self.layer226(out224)
out227 = self.layer227(out226, out2)
out228 = self.layer228(out227)
out229 = self.layer229(out228)
out229 = out229 + out226
out231 = self.layer231(out229)
out232 = self.layer232(out231)
out233 = self.layer233(out232)
out234 = self.layer234(out233)
out234 = out234 + out231
out236 = self.layer236(out234)
out237 = self.layer237(out236, out2)
out238 = self.layer238(out237)
out239 = self.layer239(out238)
out239 = out239 + out236
out241 = self.layer241(out239)
out242 = self.layer242(out241)
out243 = self.layer243(out242)
out244 = self.layer244(out243)
out244 = out244 + out241
out246 = self.layer246(out244)
out247 = self.layer247(out246, out2)
out248 = self.layer248(out247)
out249 = self.layer249(out248)
out249 = out249 + out246
out251 = self.layer251(out249)
out252 = self.layer252(out251)
out253 = self.layer253(out252)
out254 = self.layer254(out253)
out254 = out254 + out251
out256 = self.layer256(out254)
out257 = self.layer257(out256, out2)
out258 = self.layer258(out257)
out259 = self.layer259(out258)
out259 = out259 + out256
out261 = self.layer261(out259)
out262 = self.layer262(out261)
out263 = self.layer263(out262)
out264 = self.layer264(out263)
out264 = out264 + out261
out266 = self.layer266(out264)
out267 = self.layer267(out266, out2)
out268 = self.layer268(out267)
out269 = self.layer269(out268)
out269 = out269 + out266
out271 = self.layer271(out269)
out272 = self.layer272(out271)
out273 = self.layer273(out272)
out274 = self.layer274(out273)
out274 = out274 + out271
out276 = self.layer276(out274)
out277 = self.layer277(out276, out2)
out278 = self.layer278(out277)
out279 = self.layer279(out278)
out279 = out279 + out276
out281 = self.layer281(out279)
out282 = self.layer282(out281)
out283 = self.layer283(out282)
out284 = self.layer284(out283)
out284 = out284 + out281
out286 = self.layer286(out284)
out287 = self.layer287(out286, out2)
out288 = self.layer288(out287)
out289 = self.layer289(out288)
out289 = out289 + out286
out291 = self.layer291(out289)
out292 = self.layer292(out291)
out293 = self.layer293(out292)
out294 = self.layer294(out293)
out294 = out294 + out291
out296 = self.layer296(out294)
out297 = self.layer297(out296, out2)
out298 = self.layer298(out297)
out299 = self.layer299(out298)
out299 = out299 + out296
out301 = self.layer301(out299)
out302 = self.layer302(out301)
out303 = self.layer303(out302)
out304 = self.layer304(out303)
out304 = out304 + out301
out306 = self.layer306(out304)
out307 = self.layer307(out306, out2)
out308 = self.layer308(out307)
out309 = self.layer309(out308)
out309 = out309 + out306
out311 = self.layer311(out309)
out312 = self.layer312(out311)
out313 = self.layer313(out312)
out314 = self.layer314(out313)
out314 = out314 + out311
out316 = self.layer316(out314)
out317 = self.layer317(out316, out2)
out318 = self.layer318(out317)
out319 = self.layer319(out318)
out319 = out319 + out316
out321 = self.layer321(out319)
out322 = self.layer322(out321)
out323 = self.layer323(out322)
out324 = self.layer324(out323)
out324 = out324 + out321
out326 = self.layer326(out324)
out327 = self.layer327(out326, out2)
out328 = self.layer328(out327)
out329 = self.layer329(out328)
out329 = out329 + out326
out331 = self.layer331(out329)
out332 = self.layer332(out331)
out333 = self.layer333(out332)
out334 = self.layer334(out333)
out334 = out334 + out331
out336 = self.layer336(out334)
out337 = self.layer337(out336, out2)
out338 = self.layer338(out337)
out339 = self.layer339(out338)
out339 = out339 + out336
out341 = self.layer341(out339)
out342 = self.layer342(out341)
out343 = self.layer343(out342)
out344 = self.layer344(out343)
out344 = out344 + out341
out346 = self.layer346(out344)
out347 = self.layer347(out346, out2)
out348 = self.layer348(out347)
out349 = self.layer349(out348)
out349 = out349 + out346
out351 = self.layer351(out349)
out352 = self.layer352(out351)
out353 = self.layer353(out352)
out354 = self.layer354(out353)
out354 = out354 + out351
out356 = self.layer356(out354)
out357 = self.layer357(out356, out2)
out358 = self.layer358(out357)
out359 = self.layer359(out358)
out359 = out359 + out356
out361 = self.layer361(out359)
out362 = self.layer362(out361)
out363 = self.layer363(out362)
out364 = self.layer364(out363)
out364 = out364 + out361
out366 = self.layer366(out364)
out367 = self.layer367(out366, out2)
out368 = self.layer368(out367)
out369 = self.layer369(out368)
out369 = out369 + out366
out371 = self.layer371(out369)
out372 = self.layer372(out371)
out373 = self.layer373(out372)
out374 = self.layer374(out373)
out374 = out374 + out371
out376 = self.layer376(out374)
out377 = self.layer377(out376, out2)
out378 = self.layer378(out377)
out379 = self.layer379(out378)
out379 = out379 + out376
out381 = self.layer381(out379)
out382 = self.layer382(out381)
out383 = self.layer383(out382)
out384 = self.layer384(out383)
out384 = out384 + out381
out386 = self.layer386(out384)
out387 = self.layer387(out386, out2)
out388 = self.layer388(out387)
out389 = self.layer389(out388)
out389 = out389 + out386
out391 = self.layer391(out389)
out392 = self.layer392(out391)
out393 = self.layer393(out392)
out394 = self.layer394(out393)
out394 = out394 + out391
out396 = self.layer396(out394)
out397 = self.layer397(out396, out2)
out398 = self.layer398(out397)
out399 = self.layer399(out398)
out399 = out399 + out396
out401 = self.layer401(out399)
out402 = self.layer402(out401)
out403 = self.layer403(out402)
out404 = self.layer404(out403)
out404 = out404 + out401
out406 = self.layer406(out404)
out407 = self.layer407(out406, out2)
out408 = self.layer408(out407)
out409 = self.layer409(out408)
out409 = out409 + out406
out411 = self.layer411(out409)
out412 = self.layer412(out411)
out413 = self.layer413(out412)
out414 = self.layer414(out413)
out414 = out414 + out411
out416 = self.layer416(out414)
out417 = self.layer417(out416, out2)
out418 = self.layer418(out417)
out419 = self.layer419(out418)
out419 = out419 + out416
out421 = self.layer421(out419)
out422 = self.layer422(out421)
out423 = self.layer423(out422)
out424 = self.layer424(out423)
out424 = out424 + out421
out426 = self.layer426(out424)
out427 = self.layer427(out426, out2)
out428 = self.layer428(out427)
out429 = self.layer429(out428)
out429 = out429 + out426
out431 = self.layer431(out429)
out432 = self.layer432(out431)
out433 = self.layer433(out432)
out434 = self.layer434(out433)
out434 = out434 + out431
out436 = self.layer436(out434)
out437 = self.layer437(out436, out2)
out438 = self.layer438(out437)
out439 = self.layer439(out438)
out439 = out439 + out436
out441 = self.layer441(out439)
out442 = self.layer442(out441)
out443 = self.layer443(out442)
out444 = self.layer444(out443)
out444 = out444 + out441
out446 = self.layer446(out444)
out447 = self.layer447(out446, out2)
out448 = self.layer448(out447)
out449 = self.layer449(out448)
out449 = out449 + out446
out451 = self.layer451(out449)
out452 = self.layer452(out451)
out453 = self.layer453(out452)
out454 = self.layer454(out453)
out454 = out454 + out451
out456 = self.layer456(out454)
out457 = self.layer457(out456, out2)
out458 = self.layer458(out457)
out459 = self.layer459(out458)
out459 = out459 + out456
out461 = self.layer461(out459)
out462 = self.layer462(out461)
out463 = self.layer463(out462)
out464 = self.layer464(out463)
out464 = out464 + out461
out466 = self.layer466(out464)
out467 = self.layer467(out466, out2)
out468 = self.layer468(out467)
out469 = self.layer469(out468)
out469 = out469 + out466
out471 = self.layer471(out469)
out472 = self.layer472(out471)
out473 = self.layer473(out472)
out474 = self.layer474(out473)
out474 = out474 + out471
out476 = self.layer476(out474)
out477 = self.layer477(out476, out2)
out478 = self.layer478(out477)
out479 = self.layer479(out478)
out479 = out479 + out476
out481 = self.layer481(out479)
out482 = self.layer482(out481)
out483 = self.layer483(out482)
out484 = self.layer484(out483)
out484 = out484 + out481
out486 = self.layer486(out484)
out487 = self.layer487(out486)
out488 = self.layer488(out487)
out489 = self.layer489(out488)
out490 = self.layer490(out489)'''