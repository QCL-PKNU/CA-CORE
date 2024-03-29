OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
rz(pi/4) q[0];
rz(2.542405244287177) q[1];
rz(0) q[2];
sx q[2];
rz(4.198541386745212) q[2];
sx q[2];
rz(3*pi) q[2];
rz(3.101405124974904) q[2];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(3.5035527884270836) q[4];
cx q[3],q[4];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(-3.101405124974904) q[3];
cx q[2],q[3];
rz(2.2341899025028646) q[2];
rz(3.101405124974904) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(4.5974197877043945) q[5];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[10];
sx q[10];
rz(3.930451703573483) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[1],q[11];
rz(-2.542405244287177) q[11];
cx q[1],q[11];
cx q[1],q[4];
rz(2.542405244287177) q[11];
sx q[11];
cx q[4],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[12],q[6];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[6];
sx q[6];
rz(3.5667053410188396) q[6];
sx q[6];
rz(3*pi) q[6];
rz(0) q[13];
sx q[13];
rz(5.330784672576249) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[9],q[13];
rz(0) q[13];
sx q[13];
rz(0.9524006346033369) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[9],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-0.9036317484110832) q[9];
rz(0.5825827197475594) q[14];
sx q[14];
rz(6.223136717151377) q[14];
sx q[14];
rz(14.93913413267228) q[14];
rz(0) q[14];
sx q[14];
rz(5.45620912382743) q[14];
sx q[14];
rz(3*pi) q[14];
rz(-1.6426019837128982) q[14];
cx q[2],q[14];
rz(-2.2341899025028646) q[14];
sx q[14];
rz(3.0051325344052278) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[2],q[14];
rz(0) q[14];
sx q[14];
rz(3.2780527727743585) q[14];
sx q[14];
rz(13.301569846985142) q[14];
rz(3.541473576329732) q[14];
rz(-pi/2) q[14];
rz(3.9949670492105067) q[2];
sx q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[0],q[16];
rz(-pi/4) q[16];
cx q[0],q[16];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[13],q[0];
rz(3.106434608346858) q[0];
cx q[13],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.020976362309375524) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[3];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[17],q[10];
rz(0) q[10];
sx q[10];
rz(2.352733603606103) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[17],q[10];
rz(1.4855137608335434) q[10];
rz(pi/2) q[10];
rz(2.2596365831549408) q[17];
cx q[17],q[9];
rz(-2.2596365831549408) q[9];
sx q[9];
rz(2.2757917982374325) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[17],q[9];
rz(pi/2) q[17];
sx q[17];
rz(5.508796118514385) q[17];
sx q[17];
rz(5*pi/2) q[17];
rz(0) q[9];
sx q[9];
rz(4.007393508942154) q[9];
sx q[9];
rz(12.588046292335402) q[9];
rz(0) q[9];
sx q[9];
rz(8.101531944085172) q[9];
sx q[9];
rz(3*pi) q[9];
rz(-pi/2) q[9];
rz(1.351081648985648) q[18];
cx q[5],q[18];
rz(-4.5974197877043945) q[18];
sx q[18];
rz(1.7003352801750777) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[5],q[18];
rz(0) q[18];
sx q[18];
rz(4.5828500270045085) q[18];
sx q[18];
rz(12.671116099488126) q[18];
rz(-pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[16],q[18];
rz(6.13748000711989) q[18];
cx q[16],q[18];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0.09629312366826939) q[16];
rz(6.216494075924526) q[16];
rz(4.614853720101272) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
cx q[18],q[4];
cx q[4],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[17],q[18];
rz(6.012246953068791) q[18];
cx q[17],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[17];
rz(6.20102060797856) q[17];
cx q[18],q[17];
rz(-0.6146266131858705) q[17];
rz(pi/2) q[17];
rz(pi) q[18];
rz(0) q[18];
sx q[18];
rz(5.781087652911831) q[18];
sx q[18];
rz(3*pi) q[18];
rz(pi/4) q[4];
cx q[4],q[3];
rz(-pi/4) q[3];
cx q[4],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[14];
rz(pi/2) q[14];
rz(4.7338830970952195) q[14];
rz(4.771701531173994) q[3];
rz(1.1833645862594873) q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
cx q[5],q[11];
rz(-pi/4) q[11];
rz(pi/4) q[11];
cx q[11],q[13];
rz(-pi/4) q[13];
cx q[11],q[13];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[4];
rz(0) q[13];
sx q[13];
rz(0.8474195656993602) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[4];
sx q[4];
rz(0.8474195656993602) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[13],q[4];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(3.4760438233008113) q[13];
sx q[13];
rz(7.3125921741148545) q[13];
sx q[13];
rz(14.538802167308663) q[13];
rz(-pi/2) q[4];
rz(-1.1833645862594873) q[4];
x q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[1],q[5];
rz(6.205786506700941) q[5];
cx q[1],q[5];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[9];
rz(4.011296155980329) q[1];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[0];
rz(3.6688323197856887) q[0];
cx q[5],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[9];
rz(5.555802705215949) q[9];
rz(pi/2) q[9];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(2.5535339005684845) q[19];
cx q[7],q[19];
rz(-2.5535339005684845) q[19];
cx q[7],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[12];
rz(0.7286924398690824) q[12];
cx q[19],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
rz(2.066175024948277) q[19];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[10];
rz(0) q[10];
sx q[10];
rz(1.7603234023748928) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[7];
sx q[7];
rz(1.7603234023748928) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[10];
rz(-pi/2) q[10];
rz(-1.4855137608335434) q[10];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(2.919816184218235) q[7];
cx q[7],q[10];
rz(-2.919816184218235) q[10];
cx q[7],q[10];
rz(2.919816184218235) q[10];
rz(pi/2) q[10];
rz(0.6258190208727914) q[7];
cx q[2],q[7];
rz(-3.9949670492105067) q[7];
sx q[7];
rz(0.4073431380948769) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[2],q[7];
rz(5.726101786756416) q[2];
rz(1.0008822206029995) q[2];
sx q[2];
rz(9.204074900312154) q[2];
sx q[2];
rz(14.648885690896726) q[2];
rz(2.789817089834428) q[2];
rz(0) q[7];
sx q[7];
rz(5.875842169084709) q[7];
sx q[7];
rz(12.793925989107095) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[20];
cx q[20],q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
x q[20];
rz(pi/2) q[20];
cx q[15],q[20];
cx q[20],q[15];
rz(5.921947287158975) q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
id q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(0.5688727444710328) q[20];
cx q[20],q[5];
rz(-0.5688727444710328) q[5];
cx q[20],q[5];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/4) q[20];
rz(0.5688727444710328) q[5];
cx q[5],q[4];
rz(2.9079449994384206) q[4];
cx q[5],q[4];
rz(4.246542493170285) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[8],q[21];
rz(4.977272935936012) q[21];
cx q[8],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi) q[21];
id q[21];
rz(pi/4) q[21];
sx q[21];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
cx q[8],q[6];
rz(0) q[6];
sx q[6];
rz(2.7164799661607466) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[8],q[6];
rz(0) q[6];
sx q[6];
rz(6.758302956890088) q[6];
sx q[6];
rz(3*pi) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[10];
cx q[10],q[6];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-2.254814489061328) q[10];
cx q[1],q[10];
rz(-4.011296155980329) q[10];
sx q[10];
rz(0.8234214116230412) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[1],q[10];
cx q[1],q[20];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[10];
sx q[10];
rz(5.459763895556545) q[10];
sx q[10];
rz(15.690888605811036) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[15];
rz(3.019905756090315) q[15];
cx q[10],q[15];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[16];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[3];
rz(2.0048642760649504) q[16];
cx q[10],q[16];
rz(pi) q[10];
rz(0.5702494148748618) q[10];
sx q[10];
rz(8.401760361653235) q[10];
sx q[10];
rz(8.854528545894517) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(-1.4984465165243974) q[20];
rz(2.2572805183898605) q[3];
cx q[15],q[3];
id q[15];
rz(3.4874835800922153) q[15];
rz(5.365107272791006) q[3];
rz(pi/2) q[3];
cx q[4],q[20];
rz(-4.246542493170285) q[20];
sx q[20];
rz(1.6074080841083378) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[4],q[20];
rz(0) q[20];
sx q[20];
rz(4.675777223071249) q[20];
sx q[20];
rz(15.16976697046406) q[20];
cx q[4],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[4];
cx q[4],q[1];
rz(-pi/4) q[1];
cx q[4],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.6435347389090333) q[1];
cx q[15],q[1];
rz(-3.4874835800922153) q[1];
sx q[1];
rz(1.3798412626400063) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[15],q[1];
rz(0) q[1];
sx q[1];
rz(4.90334404453958) q[1];
sx q[1];
rz(11.26872680195256) q[1];
rz(pi) q[1];
x q[1];
rz(-0.23758927454447815) q[1];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(3.0667283272176618) q[15];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
rz(pi/2) q[6];
cx q[7],q[6];
cx q[6],q[7];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[11],q[6];
cx q[6],q[11];
cx q[11],q[6];
cx q[13],q[11];
cx q[11],q[13];
cx q[13],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(4.253468919859472) q[6];
sx q[6];
rz(7.271749530710832) q[6];
rz(3.1319845636534067) q[6];
rz(2.628882018526273) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[12];
rz(4.042177283647314) q[12];
cx q[8],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
id q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(-pi/4) q[12];
cx q[0],q[12];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[17];
rz(0) q[0];
sx q[0];
rz(3.3101290317731564) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[12],q[7];
rz(0) q[17];
sx q[17];
rz(2.97305627540643) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[0],q[17];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[17];
rz(0.6146266131858705) q[17];
cx q[17],q[2];
rz(-2.789817089834428) q[2];
cx q[17],q[2];
cx q[17],q[18];
rz(5.397751770891322) q[18];
cx q[17],q[18];
rz(-pi/4) q[17];
rz(2.2868816673667407) q[17];
rz(0) q[17];
sx q[17];
rz(6.7417922415879215) q[17];
sx q[17];
rz(3*pi) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[18];
rz(6.012551974050976) q[2];
sx q[2];
rz(7.582499963100894) q[2];
sx q[2];
rz(11.122757972046973) q[2];
cx q[2],q[4];
rz(pi/4) q[2];
rz(5.739577691008268) q[2];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(3.6078071625173926) q[4];
rz(-pi/4) q[7];
cx q[12],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[5];
rz(5.041947082274666) q[5];
cx q[7],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi) q[5];
rz(pi/2) q[7];
cx q[11],q[7];
cx q[7],q[11];
rz(1.2303954221372797) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(9.374603087745891) q[7];
sx q[7];
rz(5*pi/2) q[7];
rz(pi/4) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(4.889610304542773) q[8];
rz(2.2287976415830735) q[8];
cx q[8],q[19];
rz(-2.2287976415830735) q[19];
sx q[19];
rz(0.5251483298987609) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[8],q[19];
rz(0) q[19];
sx q[19];
rz(5.758036977280826) q[19];
sx q[19];
rz(9.587400577404177) q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[9];
rz(0) q[19];
sx q[19];
rz(1.6606891292747006) q[19];
sx q[19];
rz(3*pi) q[19];
rz(pi/2) q[8];
cx q[8],q[21];
x q[21];
rz(-pi/4) q[21];
cx q[21],q[16];
cx q[16],q[21];
cx q[21],q[5];
cx q[5],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[21];
rz(-pi/4) q[21];
cx q[7],q[21];
rz(pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(8.833207421213743) q[21];
sx q[21];
rz(5*pi/2) q[21];
rz(pi/4) q[21];
sx q[21];
id q[21];
rz(2.76274192075103) q[21];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
x q[8];
rz(-0.9908650779296364) q[8];
sx q[8];
rz(6.7527888845312045) q[8];
sx q[8];
rz(10.415643038699017) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[14],q[8];
rz(pi/4) q[14];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[14],q[8];
rz(-pi/4) q[8];
cx q[14],q[8];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
rz(4.43500713683688) q[14];
cx q[13],q[14];
rz(2.1749781975125417) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
cx q[16],q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[6];
rz(0.24758581154618592) q[6];
cx q[16],q[6];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
cx q[16],q[5];
rz(-pi/4) q[5];
cx q[16],q[5];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(4.504955113473279) q[5];
sx q[5];
rz(6.307577571291133) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[6],q[7];
rz(-pi/4) q[7];
cx q[6],q[7];
rz(-3.594313462502876) q[6];
rz(pi/2) q[6];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[15],q[7];
rz(-3.0667283272176618) q[7];
cx q[15],q[7];
rz(3.0667283272176618) q[7];
rz(0) q[7];
sx q[7];
rz(4.2655738884771885) q[7];
sx q[7];
rz(3*pi) q[7];
rz(pi/2) q[7];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi) q[8];
x q[8];
rz(0.10677915509686708) q[8];
cx q[4],q[8];
rz(-3.6078071625173926) q[8];
sx q[8];
rz(1.7401873185795542) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[4],q[8];
rz(0.33879508135594516) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.9403561496634647) q[4];
rz(0) q[8];
sx q[8];
rz(4.542997988600032) q[8];
sx q[8];
rz(12.925805968189906) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(1.6606891292747006) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[19],q[9];
rz(-pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
x q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[0],q[19];
rz(5.177298213080561) q[19];
cx q[0],q[19];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
id q[0];
rz(pi) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
cx q[19],q[18];
rz(pi/2) q[18];
rz(-pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[19];
rz(5.514253149978308) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[8],q[18];
rz(5.199992861802843) q[18];
cx q[8],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[19];
rz(0) q[18];
sx q[18];
rz(5.923351751572139) q[18];
sx q[18];
rz(3*pi) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
x q[19];
rz(0.5006015795366372) q[19];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[9];
rz(-5.555802705215949) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(1.2823659879700922) q[9];
cx q[9],q[12];
rz(-1.2823659879700922) q[12];
cx q[9],q[12];
rz(1.2823659879700922) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[3];
rz(0) q[12];
sx q[12];
rz(2.1103052975377987) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[3];
sx q[3];
rz(2.1103052975377987) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[12],q[3];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[3];
rz(-5.365107272791006) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[12],q[3];
rz(pi/4) q[12];
cx q[12],q[10];
rz(-pi/4) q[10];
cx q[12],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[0],q[10];
rz(3.9632098348821776) q[10];
cx q[0],q[10];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(-4.551350951507017) q[12];
rz(pi/2) q[12];
cx q[2],q[0];
rz(1.0816316541533963) q[0];
cx q[2],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.35098263863843937) q[2];
sx q[2];
rz(6.84695909731206) q[2];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/4) q[3];
rz(0.19874599453142633) q[3];
rz(0.9472985766304294) q[3];
cx q[3],q[1];
rz(-0.9472985766304294) q[1];
sx q[1];
rz(2.123852942518095) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[3],q[1];
rz(0) q[1];
sx q[1];
rz(4.159332364661491) q[1];
sx q[1];
rz(10.609665811944287) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[6];
rz(0) q[1];
sx q[1];
rz(5.381429087775565) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi/4) q[3];
cx q[3],q[18];
rz(0) q[18];
sx q[18];
rz(0.35983355560744723) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[3],q[18];
rz(3.7165531388905246) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(1.1068171441902408) q[3];
rz(0) q[6];
sx q[6];
rz(0.9017562194040214) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[1],q[6];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi) q[1];
x q[1];
rz(0.12602003908935155) q[1];
sx q[1];
rz(7.240952060554367) q[1];
sx q[1];
rz(9.298757921680028) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[6];
rz(3.594313462502876) q[6];
rz(0) q[9];
sx q[9];
rz(4.085040537613632) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[20],q[9];
rz(0) q[9];
sx q[9];
rz(2.198144769565954) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[20],q[9];
rz(4.570408525110401) q[20];
rz(4.019097753811141) q[20];
cx q[20],q[13];
rz(-4.019097753811141) q[13];
sx q[13];
rz(2.0175209162023613) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[20],q[13];
rz(0) q[13];
sx q[13];
rz(4.265664390977225) q[13];
sx q[13];
rz(11.268897517067979) q[13];
cx q[13],q[14];
rz(pi/4) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[13],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
id q[13];
cx q[13],q[16];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-0.7126502509142261) q[14];
sx q[14];
rz(2.6722207183312876) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.862233252456379) q[16];
rz(-0.5636770065024104) q[20];
sx q[20];
rz(3.6709099236450737) q[20];
rz(-pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[12];
rz(0) q[12];
sx q[12];
rz(0.4922808215875616) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[20];
sx q[20];
rz(5.790904485592025) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[20],q[12];
rz(-pi/2) q[12];
rz(4.551350951507017) q[12];
cx q[12],q[4];
rz(-pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[10],q[20];
rz(-pi/4) q[20];
cx q[10],q[20];
cx q[10],q[0];
rz(2.783354248880095) q[0];
cx q[10],q[0];
sx q[10];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[15],q[20];
cx q[20],q[15];
cx q[15],q[0];
rz(pi) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[19],q[20];
rz(-0.5006015795366372) q[20];
cx q[19],q[20];
rz(pi) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0.5006015795366372) q[20];
rz(2.2222718594012836) q[20];
rz(-2.9403561496634647) q[4];
cx q[12],q[4];
cx q[12],q[8];
rz(pi/4) q[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[13],q[4];
cx q[4],q[13];
rz(3.2511028212050075) q[13];
sx q[13];
rz(6.209477727225772) q[13];
rz(1.1258286819455483) q[13];
cx q[13],q[0];
rz(-1.1258286819455483) q[0];
cx q[13],q[0];
rz(1.1258286819455483) q[0];
rz(-2.790534644493727) q[0];
sx q[0];
rz(7.64517693925743) q[0];
sx q[0];
rz(12.215312605263106) q[0];
rz(0.9574405079502732) q[13];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.5755793470279693) q[4];
cx q[6],q[14];
rz(2.4322353670608226) q[14];
cx q[6],q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[18],q[14];
cx q[14],q[18];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(2.36082309909071) q[18];
rz(1.4130519994310424) q[6];
cx q[6],q[3];
rz(-1.4130519994310424) q[3];
sx q[3];
rz(1.73540689099872) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[6],q[3];
rz(0) q[3];
sx q[3];
rz(4.547778416180867) q[3];
sx q[3];
rz(9.73101281601018) q[3];
cx q[3],q[6];
rz(4.847248010691371) q[6];
cx q[3],q[6];
rz(0) q[3];
sx q[3];
rz(4.5518834984743775) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[21],q[3];
rz(0) q[3];
sx q[3];
rz(1.7313018087052088) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[21],q[3];
rz(-pi/2) q[21];
cx q[7],q[10];
cx q[20],q[10];
rz(-2.2222718594012836) q[10];
cx q[20],q[10];
rz(2.2222718594012836) q[10];
rz(pi/2) q[10];
rz(0.2113062704765557) q[10];
x q[20];
cx q[18],q[20];
cx q[20],q[18];
rz(0.7188812820269324) q[18];
x q[7];
x q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[14],q[7];
rz(-pi/4) q[7];
cx q[14],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[10];
rz(-0.2113062704765557) q[10];
cx q[7],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
sx q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[12],q[8];
rz(-pi/4) q[8];
cx q[12],q[8];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[2],q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
x q[12];
rz(-1.3583725893880252) q[12];
sx q[12];
rz(7.6730463258624395) q[12];
sx q[12];
rz(10.783150550157405) q[12];
cx q[14],q[12];
cx q[12],q[14];
cx q[14],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
rz(5.648385295715538) q[12];
cx q[10],q[12];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(3.23575492039493) q[12];
rz(pi/2) q[12];
rz(5.184482927584494) q[14];
sx q[14];
rz(6.307877878686281) q[14];
sx q[14];
rz(14.323094393985563) q[14];
rz(2.1168282402655807) q[14];
rz(0.07819198856811475) q[2];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[4],q[8];
rz(-2.5755793470279693) q[8];
cx q[4],q[8];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.5755793470279693) q[8];
rz(1.9569593348979302) q[8];
sx q[8];
rz(6.003824385640517) q[8];
sx q[8];
rz(10.69345995504213) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[6],q[8];
rz(5.77889563969826) q[8];
cx q[6],q[8];
cx q[6],q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(2.098088704283711) q[15];
cx q[18],q[15];
rz(-2.098088704283711) q[15];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(4.56144661837781) q[18];
rz(pi/2) q[6];
cx q[6],q[7];
x q[6];
cx q[6],q[14];
rz(-2.1168282402655807) q[14];
cx q[6],q[14];
rz(-1.3965211570276888) q[14];
cx q[18],q[14];
rz(-4.56144661837781) q[14];
sx q[14];
rz(1.1744028732920926) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[18],q[14];
rz(0) q[14];
sx q[14];
rz(5.108782433887494) q[14];
sx q[14];
rz(15.38274573617488) q[14];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(4.799395928022136) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[12];
rz(0) q[12];
sx q[12];
rz(2.542642473765867) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[8];
sx q[8];
rz(2.542642473765867) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[12];
rz(-pi/2) q[12];
rz(-3.23575492039493) q[12];
cx q[12],q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi) q[9];
cx q[9],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0) q[11];
sx q[11];
rz(3.802062267602566) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[11],q[5];
rz(pi/4) q[11];
rz(3.643273372307277) q[5];
rz(pi/4) q[9];
cx q[9],q[17];
rz(-pi/4) q[17];
cx q[9],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[16],q[17];
rz(-2.862233252456379) q[17];
cx q[16],q[17];
rz(-pi/2) q[16];
rz(0) q[16];
sx q[16];
rz(4.556727064756356) q[16];
sx q[16];
rz(3*pi) q[16];
rz(2.862233252456379) q[17];
rz(-0.3349305785887764) q[17];
cx q[5],q[17];
rz(-3.643273372307277) q[17];
sx q[17];
rz(2.661486148451961) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[5],q[17];
rz(0) q[17];
sx q[17];
rz(3.6216991587276253) q[17];
sx q[17];
rz(13.402981911665433) q[17];
cx q[17],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[1];
rz(4.487702576843056) q[1];
cx q[11],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(3.293014930219136) q[11];
sx q[11];
rz(3.9239646938972403) q[11];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[19];
rz(5.702651211566374) q[19];
cx q[17],q[19];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[21];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(4.048554104149523) q[19];
sx q[19];
rz(8.822964080829568) q[19];
sx q[19];
rz(11.944503035032772) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[2],q[5];
cx q[20],q[11];
cx q[11],q[20];
cx q[20],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[20];
rz(pi/2) q[21];
cx q[4],q[17];
cx q[17],q[4];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(1.974661145769534) q[17];
rz(pi) q[17];
x q[17];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-0.07819198856811475) q[5];
cx q[2],q[5];
rz(-pi/2) q[2];
rz(0) q[2];
sx q[2];
rz(4.235914368681554) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[3],q[2];
rz(0) q[2];
sx q[2];
rz(2.047270938498032) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[3],q[2];
rz(6.140566661341954) q[2];
sx q[2];
rz(4.593118980064991) q[2];
sx q[2];
rz(10.200790072704665) q[2];
rz(pi) q[3];
rz(pi/4) q[3];
cx q[3],q[11];
rz(-pi/4) q[11];
cx q[3],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi) q[11];
x q[11];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.07819198856811475) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0.18489421271963624) q[5];
cx q[13],q[5];
rz(-0.18489421271963624) q[5];
cx q[13],q[5];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[4],q[13];
rz(3.294437148909384) q[13];
cx q[4],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
id q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
x q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[5];
rz(3.4471911713778365) q[5];
cx q[7],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(4.196061941168073) q[7];
rz(pi/2) q[7];
cx q[6],q[7];
rz(0) q[6];
sx q[6];
rz(1.577803200272066) q[6];
sx q[6];
rz(3*pi) q[6];
rz(0) q[7];
sx q[7];
rz(1.577803200272066) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[6],q[7];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
rz(-4.196061941168073) q[7];
cx q[8],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
x q[9];
rz(-0.8869990988461809) q[9];
sx q[9];
rz(3.9689303298960095) q[9];
sx q[9];
rz(10.31177705961556) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[16];
rz(0.5061752408510827) q[16];
cx q[0],q[16];
rz(-0.5061752408510827) q[16];
cx q[0],q[16];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-5.0929538591259735) q[16];
rz(pi/2) q[16];
cx q[0],q[16];
rz(0) q[0];
sx q[0];
rz(5.3519359403505185) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[16];
sx q[16];
rz(0.9312493668290678) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[0],q[16];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(3.094417957499024) q[0];
rz(-pi/2) q[16];
rz(5.0929538591259735) q[16];
cx q[16],q[20];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[2],q[0];
rz(-3.094417957499024) q[0];
cx q[2],q[0];
cx q[0],q[17];
cx q[17],q[0];
cx q[0],q[17];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[19];
rz(3.666820529958981) q[19];
cx q[2],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[20];
rz(-pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[9],q[1];
rz(0.3605682481062296) q[1];
sx q[1];
rz(3.844917280335859) q[1];
sx q[1];
rz(15.443096262703316) q[1];
cx q[13],q[1];
rz(0.18721837189630713) q[1];
cx q[13],q[1];
rz(pi/2) q[1];
rz(6.248374119321193) q[13];
rz(pi/2) q[13];
cx q[20],q[13];
rz(0) q[13];
sx q[13];
rz(0.1329942143375482) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[20];
sx q[20];
rz(0.1329942143375482) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[20],q[13];
rz(-pi/2) q[13];
rz(-6.248374119321193) q[13];
rz(-pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
cx q[21],q[9];
cx q[9],q[21];
cx q[21],q[9];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[4];
rz(0.5302182267941442) q[4];
cx q[21],q[4];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[9];
cx q[10],q[9];
rz(2.223727636583571) q[10];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[16],q[9];
rz(2.295993620206818) q[9];
cx q[16],q[9];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
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
