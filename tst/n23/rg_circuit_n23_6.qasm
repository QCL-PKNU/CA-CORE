OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rz(0.5357705514334138) q[0];
rz(pi/2) q[0];
rz(pi/4) q[2];
rz(0.29805664398334497) q[3];
sx q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[6];
rz(0.8825710061037731) q[6];
cx q[9],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
id q[6];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi) q[9];
x q[9];
rz(pi/2) q[9];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[0];
rz(0) q[0];
sx q[0];
rz(2.853253138962708) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[10];
sx q[10];
rz(2.853253138962708) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[10],q[0];
rz(-pi/2) q[0];
rz(-0.5357705514334138) q[0];
rz(pi) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(2.259322144511082) q[10];
rz(2.7885835684464935) q[11];
sx q[11];
rz(6.095782117539291) q[11];
rz(6.235293590897426) q[11];
sx q[11];
rz(4.501077547372123) q[11];
sx q[11];
rz(9.510115775932919) q[11];
sx q[11];
cx q[9],q[11];
x q[9];
rz(pi) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.7225898279613485) q[12];
cx q[12],q[5];
rz(-1.7225898279613485) q[5];
cx q[12],q[5];
rz(1.8573331605492986) q[12];
rz(6.149195653467427) q[12];
rz(2.76832686840813) q[12];
rz(1.7225898279613485) q[5];
rz(pi/2) q[5];
cx q[5],q[3];
x q[5];
rz(pi) q[5];
x q[5];
cx q[13],q[4];
cx q[4],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-4.001481155962898) q[4];
sx q[4];
rz(4.032501586853033) q[4];
sx q[4];
rz(13.426259116732277) q[4];
rz(pi/2) q[4];
cx q[14],q[8];
cx q[8],q[14];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[2],q[15];
rz(-pi/4) q[15];
cx q[2],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[10],q[15];
rz(-2.259322144511082) q[15];
cx q[10],q[15];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.259322144511082) q[15];
cx q[15],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.8203071052014836) q[10];
rz(-1.1252893149904661) q[15];
sx q[15];
rz(2.0069778108842127) q[15];
cx q[2],q[7];
rz(pi/4) q[2];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[2],q[7];
rz(-pi/4) q[7];
cx q[2],q[7];
rz(1.24946825093057) q[2];
sx q[2];
rz(3.857803591895908) q[2];
sx q[2];
rz(8.175309709838809) q[2];
rz(-0.12257820475134278) q[2];
rz(pi/2) q[2];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[7];
cx q[4],q[7];
x q[4];
rz(3.0694683819288637) q[4];
sx q[4];
rz(3.885090435307819) q[4];
sx q[4];
rz(13.110222284456599) q[4];
rz(0.3705687366687276) q[4];
sx q[4];
rz(8.826931653596029) q[4];
sx q[4];
rz(9.054209224100651) q[4];
rz(1.3310676067456664) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(3.0505495399852687) q[17];
cx q[18],q[1];
cx q[1],q[18];
sx q[1];
rz(0.9263669184725523) q[1];
rz(pi/4) q[18];
cx q[18],q[13];
rz(-pi/4) q[13];
cx q[18],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[3];
rz(5.378090200508351) q[13];
rz(3.7764277642774084) q[13];
rz(pi/2) q[13];
x q[18];
rz(3.609776260098057) q[18];
cx q[18],q[10];
rz(-3.609776260098057) q[10];
sx q[10];
rz(0.8729190445054762) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[18],q[10];
rz(0) q[10];
sx q[10];
rz(5.41026626267411) q[10];
sx q[10];
rz(11.214247115665952) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.0431987571620076) q[18];
rz(1.144187460054578) q[18];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[2];
rz(0) q[2];
sx q[2];
rz(1.4224066561650364) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[3];
sx q[3];
rz(4.86077865101455) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[2];
rz(-pi/2) q[2];
rz(0.12257820475134278) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(2.410389018060364) q[3];
cx q[9],q[13];
rz(0) q[13];
sx q[13];
rz(1.6365745544152683) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[9];
sx q[9];
rz(1.6365745544152683) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[13];
rz(-pi/2) q[13];
rz(-3.7764277642774084) q[13];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
x q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi) q[19];
x q[19];
id q[19];
cx q[11],q[19];
rz(6.0560023384096136) q[19];
cx q[11],q[19];
rz(-pi/2) q[11];
rz(pi/2) q[19];
cx q[17],q[20];
rz(-3.0505495399852687) q[20];
cx q[17],q[20];
cx q[17],q[8];
rz(pi) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
cx q[17],q[3];
rz(3.0505495399852687) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/4) q[20];
cx q[14],q[20];
cx q[1],q[14];
rz(-0.9263669184725523) q[14];
cx q[1],q[14];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.9263669184725523) q[14];
rz(-0.5083534454755032) q[14];
sx q[14];
rz(4.951870776145243) q[14];
sx q[14];
rz(9.933131406244883) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(-2.410389018060364) q[3];
cx q[17],q[3];
cx q[11],q[3];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/4) q[11];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[6],q[8];
rz(5.177004605584347) q[8];
cx q[6],q[8];
rz(0) q[6];
sx q[6];
rz(8.298236692233672) q[6];
sx q[6];
rz(3*pi) q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(3.2283349737097953) q[21];
cx q[21],q[16];
rz(3.0759232580561555) q[16];
cx q[21],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[20],q[16];
rz(pi/4) q[16];
cx q[16],q[1];
rz(-pi/4) q[1];
cx q[16],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-0.14385679919792238) q[1];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[10],q[16];
rz(4.212320340540987) q[16];
cx q[10],q[16];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
x q[10];
rz(pi/4) q[10];
cx q[10],q[9];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[6];
cx q[18],q[1];
rz(-1.144187460054578) q[1];
sx q[1];
rz(0.5773892019330038) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[18],q[1];
rz(0) q[1];
sx q[1];
rz(5.705796105246582) q[1];
sx q[1];
rz(10.71282222002188) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[17],q[18];
cx q[18],q[17];
cx q[17],q[18];
rz(-pi/2) q[17];
rz(5.77147124524216) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[21];
cx q[21],q[0];
rz(-pi/4) q[0];
cx q[21],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[12],q[0];
rz(-2.76832686840813) q[0];
cx q[12],q[0];
rz(2.76832686840813) q[0];
cx q[0],q[15];
rz(4.697830224542702) q[15];
cx q[0],q[15];
id q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[3];
sx q[15];
cx q[15],q[1];
rz(0.557099555202998) q[21];
sx q[21];
rz(3.190113840019812) q[21];
sx q[21];
rz(10.289727581625618) q[21];
sx q[21];
cx q[19],q[21];
x q[19];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(2.1494217830807933) q[3];
cx q[0],q[3];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.360288173844022) q[0];
sx q[0];
rz(7.588603392992438) q[0];
sx q[0];
rz(15.502576925739223) q[0];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[5],q[20];
rz(0.940266959207746) q[20];
cx q[20],q[12];
rz(-0.940266959207746) q[12];
cx q[20],q[12];
rz(0.940266959207746) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[21];
sx q[20];
cx q[14],q[20];
x q[14];
rz(4.427657229367884) q[14];
rz(0) q[20];
sx q[20];
rz(4.41972534499447) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[19],q[20];
rz(0) q[20];
sx q[20];
rz(1.8634599621851164) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[19],q[20];
rz(pi/2) q[19];
cx q[18],q[19];
cx q[19],q[18];
rz(4.4714488407588195) q[18];
sx q[18];
rz(6.96923559887391) q[18];
rz(0) q[18];
sx q[18];
rz(5.345624292673682) q[18];
sx q[18];
rz(3*pi) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[21],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[12],q[21];
rz(0.703891048056364) q[21];
cx q[12],q[21];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.8269428886149948) q[12];
rz(-pi/4) q[12];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(0.8233125320579656) q[21];
sx q[21];
rz(3.786983715976107) q[21];
sx q[21];
rz(11.279812903454769) q[21];
cx q[3],q[20];
rz(4.187800870092746) q[20];
cx q[3],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-2.033140376135879) q[5];
sx q[5];
rz(3.6422297596584623) q[5];
sx q[5];
rz(11.457918336905259) q[5];
rz(2.5488813346685144) q[5];
cx q[5],q[13];
rz(-2.5488813346685144) q[13];
cx q[5],q[13];
rz(2.5488813346685144) q[13];
rz(pi/2) q[13];
sx q[13];
rz(8.199664895657625) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-0.49593224537334923) q[5];
cx q[14],q[5];
rz(-4.427657229367884) q[5];
sx q[5];
rz(1.4402446575184034) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[14],q[5];
cx q[15],q[14];
cx q[14],q[15];
rz(-pi/2) q[14];
cx q[15],q[19];
rz(5.164208855420373) q[19];
cx q[15],q[19];
rz(pi/2) q[19];
rz(0) q[5];
sx q[5];
rz(4.842940649661183) q[5];
sx q[5];
rz(14.348367435510612) q[5];
cx q[5],q[13];
rz(3.3211131613643716) q[13];
cx q[5],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[20];
cx q[13],q[18];
rz(0) q[18];
sx q[18];
rz(0.9375610145059041) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[13],q[18];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.5760829266672274) q[18];
cx q[18],q[3];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-0.5760829266672274) q[3];
cx q[18],q[3];
rz(0.36022353878190744) q[18];
sx q[18];
rz(4.684588767561818) q[18];
sx q[18];
rz(10.589248459563631) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.5760829266672274) q[3];
rz(5.072523079020005) q[3];
rz(pi) q[5];
rz(pi/2) q[5];
sx q[5];
rz(5.758890978529891) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[13];
rz(2.004607315397156) q[13];
cx q[5],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.8702265641118635) q[13];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[16];
rz(1.4430034289720268) q[16];
sx q[16];
rz(6.284854842383897) q[16];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
id q[6];
rz(pi) q[6];
cx q[6],q[14];
cx q[12],q[6];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(1.6583554820341229) q[6];
cx q[12],q[6];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[5],q[12];
rz(4.08908947659075) q[12];
cx q[5],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
sx q[6];
rz(-pi/4) q[9];
cx q[10],q[9];
rz(-1.725916413266088) q[10];
rz(pi/2) q[10];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[16];
rz(2.2496783860855705) q[16];
cx q[9],q[16];
rz(pi/2) q[16];
sx q[16];
rz(5.902188140253361) q[16];
sx q[16];
rz(5*pi/2) q[16];
sx q[16];
rz(5.6446522780057125) q[9];
rz(pi/2) q[9];
rz(1.6614550250578004) q[22];
sx q[22];
rz(5.813014070024257) q[22];
rz(pi) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(-pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[7];
rz(3.301812133199168) q[7];
cx q[22],q[7];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[4];
rz(0) q[22];
sx q[22];
rz(3.0328994641670253) q[22];
sx q[22];
rz(3*pi) q[22];
rz(0) q[4];
sx q[4];
rz(3.0328994641670253) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[22],q[4];
rz(-pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(-pi/2) q[4];
rz(-1.3310676067456664) q[4];
rz(-pi/2) q[4];
cx q[22],q[4];
rz(4.0990820654234374) q[22];
rz(pi/2) q[22];
rz(pi/2) q[4];
rz(0.626612054719247) q[4];
id q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(6.148970475318458) q[7];
rz(pi/2) q[7];
cx q[8],q[7];
rz(0) q[7];
sx q[7];
rz(2.5456303785816328) q[7];
sx q[7];
rz(3*pi) q[7];
rz(0) q[8];
sx q[8];
rz(2.5456303785816328) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[7];
rz(-pi/2) q[7];
rz(-6.148970475318458) q[7];
rz(3.2577816344009105) q[7];
sx q[7];
rz(7.39644712212995) q[7];
sx q[7];
rz(11.3698241886638) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[2],q[7];
rz(pi/4) q[2];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[2],q[7];
rz(-pi/4) q[7];
cx q[2],q[7];
id q[2];
rz(pi/4) q[2];
rz(5.291287342749865) q[2];
rz(-pi/2) q[2];
sx q[2];
sx q[2];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[22];
rz(0) q[22];
sx q[22];
rz(0.06959573885933112) q[22];
sx q[22];
rz(3*pi) q[22];
rz(0) q[7];
sx q[7];
rz(0.06959573885933112) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[22];
rz(-pi/2) q[22];
rz(-4.0990820654234374) q[22];
rz(0) q[22];
sx q[22];
rz(5.274659204613758) q[22];
sx q[22];
rz(3*pi) q[22];
rz(pi/2) q[22];
rz(-pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[21],q[7];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[21],q[7];
rz(5.667838345860564) q[7];
cx q[21],q[7];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(1.1582387642086236) q[21];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[22],q[7];
rz(5.811522378878823) q[7];
cx q[22],q[7];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(0.9000972682193201) q[22];
sx q[22];
rz(7.99843854246748) q[22];
sx q[22];
rz(8.52468069255006) q[22];
rz(3.2358231153089685) q[22];
sx q[22];
rz(5.588921204252156) q[22];
sx q[22];
rz(13.793693355628314) q[22];
rz(-1.457274732585404) q[22];
sx q[22];
rz(6.055233814943393) q[22];
rz(pi/2) q[22];
sx q[22];
rz(3.7492001471503213) q[22];
sx q[22];
rz(5*pi/2) q[22];
rz(5.127711855925055) q[22];
rz(0) q[22];
sx q[22];
rz(3.4406384333092834) q[22];
sx q[22];
rz(3*pi) q[22];
rz(-pi/2) q[22];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0) q[7];
sx q[7];
rz(8.639555329906516) q[7];
sx q[7];
rz(3*pi) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[12],q[7];
rz(pi) q[12];
x q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-1.2472200410678622) q[7];
sx q[7];
rz(2.1634294962502745) q[7];
rz(pi) q[7];
x q[7];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(2.354171278850268) q[8];
cx q[8],q[17];
rz(pi/2) q[17];
rz(1.8247108166777464) q[17];
cx q[17],q[1];
rz(-1.8247108166777464) q[1];
cx q[17],q[1];
rz(1.8247108166777464) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[11],q[1];
rz(-pi/4) q[1];
cx q[11],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[1];
sx q[11];
cx q[14],q[1];
x q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.329236607130292) q[14];
cx q[16],q[14];
rz(-2.329236607130292) q[14];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[17],q[0];
rz(2.2724782393604444) q[0];
cx q[17],q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[20];
rz(5.270888219150854) q[17];
cx q[19],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
x q[19];
rz(1.1875777734366721) q[19];
rz(2.9660017357736117) q[19];
cx q[16],q[19];
rz(-2.9660017357736117) q[19];
cx q[16],q[19];
rz(4.773885641976606) q[16];
sx q[16];
rz(6.893483587493794) q[16];
sx q[16];
rz(13.952763058429039) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(5.303129855188415) q[20];
cx q[0],q[20];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
id q[0];
rz(pi/4) q[0];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
id q[20];
rz(1.0489076551660494) q[20];
cx q[14],q[20];
rz(-1.0489076551660494) q[20];
cx q[14],q[20];
rz(-pi/2) q[14];
rz(-pi/2) q[20];
cx q[21],q[1];
rz(-1.1582387642086236) q[1];
cx q[21],q[1];
rz(1.1582387642086236) q[1];
rz(pi/2) q[1];
sx q[1];
rz(5.537505689128382) q[1];
sx q[1];
rz(5*pi/2) q[1];
rz(1.739690693322892) q[1];
sx q[1];
rz(7.460098013262363) q[1];
rz(0) q[1];
sx q[1];
rz(5.650458066053089) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi) q[21];
x q[21];
rz(0.030470974641719772) q[21];
sx q[21];
rz(6.435135911586629) q[21];
rz(pi) q[21];
x q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[5],q[20];
rz(pi/2) q[20];
rz(-pi/2) q[5];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[10];
rz(0) q[10];
sx q[10];
rz(0.031571743899655225) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[8];
sx q[8];
rz(6.251613563279931) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[10];
rz(-pi/2) q[10];
rz(1.725916413266088) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[15],q[10];
rz(3.1397681496183623) q[10];
cx q[15],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi) q[10];
rz(0) q[15];
sx q[15];
rz(6.057326749261398) q[15];
sx q[15];
rz(3*pi) q[15];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[9];
rz(0) q[8];
sx q[8];
rz(1.9904199895513979) q[8];
sx q[8];
rz(3*pi) q[8];
rz(0) q[9];
sx q[9];
rz(1.9904199895513979) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[8],q[9];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi) q[8];
rz(pi/2) q[8];
cx q[8],q[6];
cx q[6],q[15];
rz(4.125421199046645) q[15];
cx q[6],q[15];
rz(pi) q[6];
cx q[6],q[5];
rz(pi/2) q[5];
x q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[11];
cx q[11],q[8];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(2.5746303647096926) q[11];
cx q[11],q[15];
rz(-2.5746303647096926) q[15];
cx q[11],q[15];
rz(2.5746303647096926) q[15];
cx q[15],q[1];
rz(0) q[1];
sx q[1];
rz(0.6327272411264975) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[15],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[15];
sx q[15];
rz(3.194456071218096) q[15];
sx q[15];
rz(3*pi) q[15];
id q[8];
rz(pi/4) q[8];
cx q[8],q[16];
rz(-pi/4) q[16];
cx q[8],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-3.5943589734278634) q[16];
sx q[16];
rz(6.674790776565988) q[16];
sx q[16];
rz(13.019136934197242) q[16];
rz(2.3163018684998953) q[8];
cx q[5],q[8];
rz(-2.3163018684998953) q[8];
cx q[5],q[8];
rz(-pi/4) q[8];
rz(-pi/2) q[9];
rz(-5.6446522780057125) q[9];
rz(pi/4) q[9];
cx q[9],q[4];
rz(-pi/4) q[4];
cx q[9],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
cx q[17],q[4];
rz(2.363174481253116) q[17];
cx q[17],q[13];
rz(-2.363174481253116) q[13];
sx q[13];
rz(1.0167634524003755) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[17],q[13];
rz(0) q[13];
sx q[13];
rz(5.26642185477921) q[13];
sx q[13];
rz(10.917725877910632) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(3.4393984378475673) q[17];
sx q[17];
rz(3.723634914602477) q[17];
sx q[17];
rz(11.162737023203919) q[17];
id q[17];
rz(pi/2) q[17];
cx q[3],q[13];
rz(2.642704492107179) q[13];
cx q[3],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[3];
rz(pi/2) q[4];
rz(0) q[4];
sx q[4];
rz(3.544787089953638) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[10],q[4];
rz(0) q[4];
sx q[4];
rz(2.7383982172259484) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[10],q[4];
cx q[10],q[18];
rz(pi/4) q[10];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[10],q[18];
rz(-pi/4) q[18];
cx q[10],q[18];
rz(pi) q[10];
x q[10];
rz(1.8580257904882647) q[10];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(6.448086058271621) q[18];
sx q[18];
rz(5*pi/2) q[18];
rz(-pi/4) q[18];
rz(pi) q[4];
x q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
cx q[13],q[4];
rz(pi/2) q[13];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
cx q[4],q[15];
rz(0) q[15];
sx q[15];
rz(3.08872923596149) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[4],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[4];
cx q[4],q[15];
rz(-pi/4) q[15];
cx q[4],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(0) q[15];
sx q[15];
rz(4.953439768463133) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0.17034232434536678) q[4];
sx q[4];
rz(5.822738780562544) q[4];
sx q[4];
rz(15.186409946281724) q[4];
rz(pi) q[4];
rz(1.038950550449778) q[4];
sx q[4];
rz(6.692838022249699) q[4];
sx q[4];
rz(8.385827410319601) q[4];
rz(2.6407502585617775) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[0],q[9];
rz(-pi/4) q[9];
cx q[0],q[9];
cx q[0],q[14];
cx q[0],q[3];
rz(pi) q[0];
x q[0];
rz(pi/2) q[14];
rz(0.37878513911273154) q[14];
cx q[20],q[14];
rz(-0.37878513911273154) q[14];
cx q[20],q[14];
cx q[14],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi) q[14];
rz(pi) q[14];
rz(pi/4) q[14];
cx q[20],q[1];
rz(2.654469092621916) q[1];
cx q[20],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.627944901901384) q[1];
cx q[1],q[0];
rz(-2.627944901901384) q[0];
cx q[1],q[0];
rz(2.627944901901384) q[0];
rz(pi/2) q[0];
rz(-1.3890839267906023) q[1];
sx q[1];
rz(5.735600599236348) q[1];
sx q[1];
rz(10.813861887559982) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[3];
cx q[3],q[21];
rz(3.7627734884004846) q[21];
cx q[3],q[21];
cx q[18],q[3];
rz(-pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[16];
rz(0.009296983269067832) q[16];
cx q[21],q[16];
rz(-1.0101628059230217) q[16];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[18];
rz(2.848257959947927) q[18];
cx q[21],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(2.860416127713815) q[3];
rz(1.5783077845241977) q[3];
cx q[5],q[12];
cx q[12],q[5];
cx q[5],q[12];
rz(pi) q[12];
x q[12];
rz(5.464558796416541) q[5];
rz(1.0133546011488896) q[5];
cx q[5],q[16];
rz(-1.0133546011488896) q[16];
sx q[16];
rz(1.5383660406413349) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[5],q[16];
rz(0) q[16];
sx q[16];
rz(4.744819266538252) q[16];
sx q[16];
rz(11.44829536784129) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
rz(-pi/4) q[16];
cx q[14],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[19],q[9];
rz(pi/4) q[19];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[19],q[9];
rz(-pi/4) q[9];
cx q[19],q[9];
rz(1.3291927385213744) q[19];
cx q[19],q[11];
rz(-1.3291927385213744) q[11];
cx q[19],q[11];
rz(1.3291927385213744) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(1.23005168350136) q[11];
rz(2.6349011791706363) q[19];
cx q[19],q[6];
cx q[3],q[11];
rz(-1.5783077845241977) q[11];
sx q[11];
rz(1.04876066381786) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[3],q[11];
rz(0) q[11];
sx q[11];
rz(5.234424643361726) q[11];
sx q[11];
rz(9.773034061792217) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-2.6349011791706363) q[6];
cx q[19],q[6];
cx q[19],q[20];
rz(pi/4) q[19];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[19],q[20];
rz(-pi/4) q[20];
cx q[19],q[20];
x q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[3],q[19];
rz(5.843442790795988) q[19];
cx q[3],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/4) q[19];
rz(0.5004461500591129) q[19];
sx q[19];
rz(4.670054675486042) q[19];
sx q[19];
rz(10.512702503879895) q[19];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.6349011791706363) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[7];
cx q[7],q[6];
rz(-pi/4) q[6];
cx q[7],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[7];
cx q[0],q[7];
x q[0];
rz(pi) q[0];
cx q[8],q[6];
cx q[6],q[8];
cx q[6],q[15];
rz(0) q[15];
sx q[15];
rz(1.3297455387164536) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[6],q[15];
rz(-pi/4) q[15];
rz(1.2411172169511897) q[15];
cx q[4],q[15];
rz(-1.2411172169511897) q[15];
cx q[4],q[15];
cx q[6],q[3];
rz(4.300518391167094) q[3];
cx q[6],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.701794779048182) q[3];
rz(2.947366848798739) q[6];
cx q[8],q[7];
rz(4.22565484342944) q[7];
cx q[8],q[7];
rz(-pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[2],q[9];
rz(1.9726411995820596) q[9];
cx q[2],q[9];
rz(0.5938218947107957) q[2];
cx q[10],q[2];
rz(-1.8580257904882647) q[2];
sx q[2];
rz(2.465946443248008) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[10],q[2];
rz(2.9841628687444883) q[10];
cx q[17],q[10];
rz(-2.9841628687444883) q[10];
cx q[17],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[20];
x q[17];
rz(0.5427343270528219) q[17];
rz(0) q[2];
sx q[2];
rz(3.8172388639315784) q[2];
sx q[2];
rz(10.688981856546848) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.1018523603269212) q[20];
cx q[10],q[20];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-1.6523534142170164) q[10];
sx q[10];
rz(4.11298191183582) q[10];
sx q[10];
rz(11.077131374986395) q[10];
rz(1.369784161291425) q[10];
cx q[10],q[3];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[11];
cx q[11],q[18];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(5.196676304884153) q[11];
sx q[11];
rz(8.69219355453367) q[11];
sx q[11];
rz(14.17989476569593) q[11];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[22],q[20];
rz(5.34603472624514) q[20];
cx q[22],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(0.14146353971029935) q[22];
rz(pi/2) q[22];
rz(-1.701794779048182) q[3];
cx q[10],q[3];
cx q[5],q[17];
rz(-0.5427343270528219) q[17];
cx q[5],q[17];
rz(3.2679125905177124) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
x q[5];
rz(1.5766071574757317) q[5];
cx q[7],q[22];
rz(0) q[22];
sx q[22];
rz(2.1586968465287355) q[22];
sx q[22];
rz(3*pi) q[22];
rz(0) q[7];
sx q[7];
rz(2.1586968465287355) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[22];
rz(-pi/2) q[22];
rz(-0.14146353971029935) q[22];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[9];
cx q[13],q[9];
x q[13];
sx q[13];
rz(4.506172937820046) q[13];
sx q[13];
rz(9.392257131617683) q[13];
sx q[13];
rz(14.480139291391879) q[13];
cx q[21],q[13];
cx q[13],q[21];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[14];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[14],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(2.0117289723962757) q[21];
cx q[21],q[8];
rz(-2.0117289723962757) q[8];
cx q[21],q[8];
rz(0) q[21];
sx q[21];
rz(4.301703355352961) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[18],q[21];
rz(0) q[21];
sx q[21];
rz(1.9814819518266253) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[18],q[21];
rz(2.0117289723962757) q[8];
cx q[8],q[6];
rz(-2.947366848798739) q[6];
cx q[8],q[6];
cx q[9],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(pi/4) q[2];
cx q[2],q[1];
rz(-pi/4) q[1];
cx q[2],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.5725756822666852) q[1];
cx q[14],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[14];
cx q[14],q[0];
rz(-pi/4) q[0];
cx q[14],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(4.031355039498978) q[2];
sx q[2];
rz(7.150951610774308) q[2];
cx q[20],q[1];
rz(-1.5725756822666852) q[1];
cx q[20],q[1];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[12],q[9];
rz(pi/4) q[12];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[12],q[9];
rz(-pi/4) q[9];
cx q[12],q[9];
rz(pi/2) q[12];
cx q[12],q[16];
x q[12];
rz(4.5662824850285135) q[12];
sx q[12];
rz(2.7301127156214564) q[12];
cx q[16],q[17];
rz(5.049829511982168) q[17];
cx q[16],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.2219311724632247) q[9];
sx q[9];
rz(6.281190326979431) q[9];
sx q[9];
rz(15.141980170776351) q[9];
rz(0.010136144890785515) q[9];
sx q[9];
rz(6.400682331029772) q[9];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];