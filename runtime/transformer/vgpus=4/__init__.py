import sys
sys.path.append("..")
from vpipe import Stage
from vpipe import Transformer

def arch():
    return "transformer"

def model(criterion, partition, recompute_ratio):
    _declares = get_declares()
    _calculations = get_caculations()
    module = Transformer(_declares, _calculations)
    module.generate_layer_blocks()
    start = 0
    inputs = []
    outputs = [['out177']]
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
        (criterion, outputs[3], ["loss"])
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
    return '''self.layer5 = SinusoidalPositionalEmbedding(1024, 1, False, 1026)
self.layer7 = SinusoidalPositionalEmbedding(1024, 1, True, 1026)
self.layer8 = torch.nn.Embedding(33712, 1024, padding_idx=1)
self.layer9 = Scale(1024)
self.layer10 = torch.nn.Embedding(33712, 1024, padding_idx=1)
self.layer11 = Scale(1024)
self.layer13 = torch.nn.Dropout(p=0.1)
self.layer15 = FusedLayerNorm(1024)
self.layer16 = DecoderAttention(1024, 16, dropout=0.1)
self.layer17 = torch.nn.Dropout(p=0.1)
self.layer19 = FusedLayerNorm(1024)
self.layer21 = torch.nn.Dropout(p=0.1)
self.layer23 = FusedLayerNorm(1024)
self.layer24 = EncoderAttention(1024, 16, dropout=0.1, static_kv=False)
self.layer25 = torch.nn.Dropout(p=0.1)
self.layer27 = FusedLayerNorm(1024)
self.layer28 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer29 = torch.nn.Threshold(threshold=0, value=0)
self.layer30 = torch.nn.Dropout(p=0.1)
self.layer31 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer32 = torch.nn.Dropout(p=0.1)
self.layer34 = FusedLayerNorm(1024)
self.layer35 = EncoderAttention(1024, 16, dropout=0.1, static_kv=False)
self.layer36 = torch.nn.Dropout(p=0.1)
self.layer38 = FusedLayerNorm(1024)
self.layer39 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer40 = torch.nn.Threshold(threshold=0, value=0)
self.layer41 = torch.nn.Dropout(p=0.1)
self.layer42 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer43 = torch.nn.Dropout(p=0.1)
self.layer45 = FusedLayerNorm(1024)
self.layer46 = EncoderAttention(1024, 16, dropout=0.1, static_kv=False)
self.layer47 = torch.nn.Dropout(p=0.1)
self.layer49 = FusedLayerNorm(1024)
self.layer50 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer51 = torch.nn.Threshold(threshold=0, value=0)
self.layer52 = torch.nn.Dropout(p=0.1)
self.layer53 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer54 = torch.nn.Dropout(p=0.1)
self.layer56 = FusedLayerNorm(1024)
self.layer57 = EncoderAttention(1024, 16, dropout=0.1, static_kv=False)
self.layer58 = torch.nn.Dropout(p=0.1)
self.layer60 = FusedLayerNorm(1024)
self.layer61 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer62 = torch.nn.Threshold(threshold=0, value=0)
self.layer63 = torch.nn.Dropout(p=0.1)
self.layer64 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer65 = torch.nn.Dropout(p=0.1)
self.layer67 = FusedLayerNorm(1024)
self.layer68 = EncoderAttention(1024, 16, dropout=0.1, static_kv=False)
self.layer69 = torch.nn.Dropout(p=0.1)
self.layer71 = FusedLayerNorm(1024)
self.layer72 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer73 = torch.nn.Threshold(threshold=0, value=0)
self.layer74 = torch.nn.Dropout(p=0.1)
self.layer75 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer76 = torch.nn.Dropout(p=0.1)
self.layer78 = FusedLayerNorm(1024)
self.layer79 = EncoderAttention(1024, 16, dropout=0.1, static_kv=False)
self.layer80 = torch.nn.Dropout(p=0.1)
self.layer82 = FusedLayerNorm(1024)
self.layer83 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer84 = torch.nn.Threshold(threshold=0, value=0)
self.layer85 = torch.nn.Dropout(p=0.1)
self.layer86 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer87 = torch.nn.Dropout(p=0.1)
self.layer89 = FusedLayerNorm(1024)
self.layer90 = EncoderAttention(1024, 16, dropout=0.1, static_kv=True)
self.layer91 = torch.nn.Dropout(p=0.1)
self.layer93 = FusedLayerNorm(1024)
self.layer94 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer95 = torch.nn.Threshold(threshold=0, value=0)
self.layer96 = torch.nn.Dropout(p=0.1)
self.layer97 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer98 = torch.nn.Dropout(p=0.1)
self.layer100 = FusedLayerNorm(1024)
self.layer101 = DecoderAttention(1024, 16, dropout=0.1)
self.layer102 = torch.nn.Dropout(p=0.1)
self.layer104 = FusedLayerNorm(1024)
self.layer105 = EncoderAttention(1024, 16, dropout=0.1, static_kv=True)
self.layer106 = torch.nn.Dropout(p=0.1)
self.layer108 = FusedLayerNorm(1024)
self.layer109 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer110 = torch.nn.Threshold(threshold=0, value=0)
self.layer111 = torch.nn.Dropout(p=0.1)
self.layer112 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer113 = torch.nn.Dropout(p=0.1)
self.layer115 = FusedLayerNorm(1024)
self.layer116 = DecoderAttention(1024, 16, dropout=0.1)
self.layer117 = torch.nn.Dropout(p=0.1)
self.layer119 = FusedLayerNorm(1024)
self.layer120 = EncoderAttention(1024, 16, dropout=0.1, static_kv=True)
self.layer121 = torch.nn.Dropout(p=0.1)
self.layer123 = FusedLayerNorm(1024)
self.layer124 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer125 = torch.nn.Threshold(threshold=0, value=0)
self.layer126 = torch.nn.Dropout(p=0.1)
self.layer127 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer128 = torch.nn.Dropout(p=0.1)
self.layer130 = FusedLayerNorm(1024)
self.layer131 = DecoderAttention(1024, 16, dropout=0.1)
self.layer132 = torch.nn.Dropout(p=0.1)
self.layer134 = FusedLayerNorm(1024)
self.layer135 = EncoderAttention(1024, 16, dropout=0.1, static_kv=True)
self.layer136 = torch.nn.Dropout(p=0.1)
self.layer138 = FusedLayerNorm(1024)
self.layer139 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer140 = torch.nn.Threshold(threshold=0, value=0)
self.layer141 = torch.nn.Dropout(p=0.1)
self.layer142 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer143 = torch.nn.Dropout(p=0.1)
self.layer145 = FusedLayerNorm(1024)
self.layer146 = DecoderAttention(1024, 16, dropout=0.1)
self.layer147 = torch.nn.Dropout(p=0.1)
self.layer149 = FusedLayerNorm(1024)
self.layer150 = EncoderAttention(1024, 16, dropout=0.1, static_kv=True)
self.layer151 = torch.nn.Dropout(p=0.1)
self.layer153 = FusedLayerNorm(1024)
self.layer154 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer155 = torch.nn.Threshold(threshold=0, value=0)
self.layer156 = torch.nn.Dropout(p=0.1)
self.layer157 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer158 = torch.nn.Dropout(p=0.1)
self.layer160 = FusedLayerNorm(1024)
self.layer161 = DecoderAttention(1024, 16, dropout=0.1)
self.layer162 = torch.nn.Dropout(p=0.1)
self.layer164 = FusedLayerNorm(1024)
self.layer165 = EncoderAttention(1024, 16, dropout=0.1, static_kv=True)
self.layer166 = torch.nn.Dropout(p=0.1)
self.layer168 = FusedLayerNorm(1024)
self.layer169 = torch.nn.Linear(in_features=1024, out_features=4096, bias=True)
self.layer170 = torch.nn.Threshold(threshold=0, value=0)
self.layer171 = torch.nn.Dropout(p=0.1)
self.layer172 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
self.layer173 = torch.nn.Dropout(p=0.1)
self.layer175 = FusedLayerNorm(1024)
self.layer177 = torch.nn.Linear(in_features=1024, out_features=33712, bias=False)'''

def get_caculations():
    return '''out7 = self.layer7(out0)
out10 = self.layer10(out0)
out11 = self.layer11(out10)
out11 = out11 + out7
out21 = self.layer21(out11)
out22 = out21.transpose(0, 1)
out23 = self.layer23(out22)
out24 = self.layer24(out23, out23, out1)
out25 = self.layer25(out24)
out25 = out25 + out22
out27 = self.layer27(out25)
out28 = self.layer28(out27)
out29 = self.layer29(out28)
out30 = self.layer30(out29)
out31 = self.layer31(out30)
out32 = self.layer32(out31)
out32 = out32 + out25
out34 = self.layer34(out32)
out35 = self.layer35(out34, out34, out1)
out36 = self.layer36(out35)
out36 = out36 + out32
out38 = self.layer38(out36)
out39 = self.layer39(out38)
out40 = self.layer40(out39)
out41 = self.layer41(out40)
out42 = self.layer42(out41)
out43 = self.layer43(out42)
out43 = out43 + out36
out45 = self.layer45(out43)
out46 = self.layer46(out45, out45, out1)
out47 = self.layer47(out46)
out47 = out47 + out43
out49 = self.layer49(out47)
out50 = self.layer50(out49)
out51 = self.layer51(out50)
out52 = self.layer52(out51)
out53 = self.layer53(out52)
out54 = self.layer54(out53)
out54 = out54 + out47
out56 = self.layer56(out54)
out57 = self.layer57(out56, out56, out1)
out58 = self.layer58(out57)
out58 = out58 + out54
out60 = self.layer60(out58)
out61 = self.layer61(out60)
out62 = self.layer62(out61)
out63 = self.layer63(out62)
out64 = self.layer64(out63)
out65 = self.layer65(out64)
out65 = out65 + out58
out67 = self.layer67(out65)
out68 = self.layer68(out67, out67, out1)
out69 = self.layer69(out68)
out69 = out69 + out65
out71 = self.layer71(out69)
out72 = self.layer72(out71)
out73 = self.layer73(out72)
out74 = self.layer74(out73)
out75 = self.layer75(out74)
out76 = self.layer76(out75)
out76 = out76 + out69
out78 = self.layer78(out76)
out79 = self.layer79(out78, out78, out1)
out80 = self.layer80(out79)
out80 = out80 + out76
out82 = self.layer82(out80)
out83 = self.layer83(out82)
out84 = self.layer84(out83)
out85 = self.layer85(out84)
out86 = self.layer86(out85)
out87 = self.layer87(out86)
out87 = out87 + out80
out89 = self.layer89(out87)
out5 = self.layer5(out2)
out8 = self.layer8(out2)
out9 = self.layer9(out8)
out9 = out9 + out5
out13 = self.layer13(out9)
out14 = out13.transpose(0, 1)
out15 = self.layer15(out14)
out16 = self.layer16(out15)
out17 = self.layer17(out16)
out17 = out17 + out14
out19 = self.layer19(out17)
out90 = self.layer90(out19, out89, out1)
out91 = self.layer91(out90)
out91 = out91 + out17
out93 = self.layer93(out91)
out94 = self.layer94(out93)
out95 = self.layer95(out94)
out96 = self.layer96(out95)
out97 = self.layer97(out96)
out98 = self.layer98(out97)
out98 = out98 + out91
out100 = self.layer100(out98)
out101 = self.layer101(out100)
out102 = self.layer102(out101)
out102 = out102 + out98
out104 = self.layer104(out102)
out105 = self.layer105(out104, out89, out1)
out106 = self.layer106(out105)
out106 = out106 + out102
out108 = self.layer108(out106)
out109 = self.layer109(out108)
out110 = self.layer110(out109)
out111 = self.layer111(out110)
out112 = self.layer112(out111)
out113 = self.layer113(out112)
out113 = out113 + out106
out115 = self.layer115(out113)
out116 = self.layer116(out115)
out117 = self.layer117(out116)
out117 = out117 + out113
out119 = self.layer119(out117)
out120 = self.layer120(out119, out89, out1)
out121 = self.layer121(out120)
out121 = out121 + out117
out123 = self.layer123(out121)
out124 = self.layer124(out123)
out125 = self.layer125(out124)
out126 = self.layer126(out125)
out127 = self.layer127(out126)
out128 = self.layer128(out127)
out128 = out128 + out121
out130 = self.layer130(out128)
out131 = self.layer131(out130)
out132 = self.layer132(out131)
out132 = out132 + out128
out134 = self.layer134(out132)
out135 = self.layer135(out134, out89, out1)
out136 = self.layer136(out135)
out136 = out136 + out132
out138 = self.layer138(out136)
out139 = self.layer139(out138)
out140 = self.layer140(out139)
out141 = self.layer141(out140)
out142 = self.layer142(out141)
out143 = self.layer143(out142)
out143 = out143 + out136
out145 = self.layer145(out143)
out146 = self.layer146(out145)
out147 = self.layer147(out146)
out147 = out147 + out143
out149 = self.layer149(out147)
out150 = self.layer150(out149, out89, out1)
out151 = self.layer151(out150)
out151 = out151 + out147
out153 = self.layer153(out151)
out154 = self.layer154(out153)
out155 = self.layer155(out154)
out156 = self.layer156(out155)
out157 = self.layer157(out156)
out158 = self.layer158(out157)
out158 = out158 + out151
out160 = self.layer160(out158)
out161 = self.layer161(out160)
out162 = self.layer162(out161)
out162 = out162 + out158
out164 = self.layer164(out162)
out165 = self.layer165(out164, out89, out1)
out166 = self.layer166(out165)
out166 = out166 + out162
out168 = self.layer168(out166)
out169 = self.layer169(out168)
out170 = self.layer170(out169)
out171 = self.layer171(out170)
out172 = self.layer172(out171)
out173 = self.layer173(out172)
out173 = out173 + out166
out175 = self.layer175(out173)
out176 = out175.transpose(0, 1)
out177 = self.layer177(out176)'''
