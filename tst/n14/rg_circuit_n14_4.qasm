OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
rz(pi/2) q[0];
rz(2.0449658067039254) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi) q[3];
x q[3];
rz(2.1379012892209848) q[3];
rz(-pi/2) q[3];
rz(-2.891571678636841) q[3];
rz(pi/2) q[3];
id q[4];
x q[4];
id q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
x q[5];
rz(1.8362117502219841) q[6];
rz(2.3298938973087733) q[7];
rz(pi/4) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(-pi/4) q[9];
cx q[8],q[9];
rz(0) q[8];
sx q[8];
rz(8.517841881546548) q[8];
sx q[8];
rz(3*pi) q[8];
sx q[8];
rz(0.005172893261070982) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0) q[9];
sx q[9];
rz(9.411565843610198) q[9];
sx q[9];
rz(3*pi) q[9];
rz(3.087305223266813) q[9];
sx q[9];
rz(5.627365441043804) q[9];
sx q[9];
rz(12.76903624831532) q[9];
rz(6.063151368801777) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.6032617128928366) q[11];
cx q[7],q[11];
rz(-2.3298938973087733) q[11];
sx q[11];
rz(0.5446252566122296) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[7],q[11];
rz(0) q[11];
sx q[11];
rz(5.738560050567356) q[11];
sx q[11];
rz(10.151410145185316) q[11];
rz(pi/2) q[11];
sx q[11];
rz(9.278466465726613) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0.9070885788379766) q[11];
rz(-pi/4) q[7];
rz(pi) q[7];
x q[7];
id q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.5616109947941552) q[12];
rz(3.806681279118209) q[12];
cx q[12],q[6];
rz(-3.806681279118209) q[6];
sx q[6];
rz(1.7827423234236348) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[12],q[6];
id q[12];
rz(0) q[6];
sx q[6];
rz(4.500442983755951) q[6];
sx q[6];
rz(11.395247489665604) q[6];
rz(-1.6952073904768987) q[6];
cx q[0],q[6];
rz(-2.0449658067039254) q[6];
sx q[6];
rz(1.9612933454909358) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[0],q[6];
id q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0) q[6];
sx q[6];
rz(4.32189196168865) q[6];
sx q[6];
rz(13.164951157950203) q[6];
rz(3.5066895283101895) q[6];
sx q[6];
rz(4.404107644083201) q[6];
sx q[6];
rz(10.617846028018533) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[9],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[9];
cx q[9],q[0];
rz(-pi/4) q[0];
cx q[9],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[8];
rz(4.5695443882568485) q[8];
cx q[0],q[8];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/4) q[0];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[13],q[1];
rz(3.461259757824685) q[1];
cx q[13],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[2];
rz(pi/4) q[13];
cx q[13],q[10];
rz(-pi/4) q[10];
cx q[13],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.836564446566438) q[10];
cx q[10],q[12];
rz(-2.836564446566438) q[12];
cx q[10],q[12];
rz(2.651149861386786) q[10];
rz(pi/2) q[10];
rz(2.836564446566438) q[12];
rz(0.05815023671324566) q[12];
cx q[13],q[11];
rz(-0.9070885788379766) q[11];
cx q[13],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[3];
rz(0) q[13];
sx q[13];
rz(3.6331296261324213) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[2],q[1];
rz(0) q[1];
sx q[1];
rz(4.354714804741894) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[1],q[11];
rz(3.372284454904188) q[11];
cx q[1],q[11];
rz(0) q[1];
sx q[1];
rz(7.3571382201178865) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi/4) q[1];
cx q[1],q[7];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[12],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[12];
cx q[12],q[11];
rz(-pi/4) q[11];
cx q[12],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(2.9473745066654335) q[11];
rz(-pi/2) q[11];
rz(4.109264915408066) q[12];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.1847228350503194) q[2];
rz(0) q[3];
sx q[3];
rz(2.650055681047165) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[13],q[3];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(0.7585779112760256) q[13];
id q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.3604584985367698) q[13];
rz(-pi/2) q[3];
rz(2.891571678636841) q[3];
cx q[4],q[10];
rz(0) q[10];
sx q[10];
rz(1.6821580275230184) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[4];
sx q[4];
rz(1.6821580275230184) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[4],q[10];
rz(-pi/2) q[10];
rz(-2.651149861386786) q[10];
rz(0) q[10];
sx q[10];
rz(9.391042438925457) q[10];
sx q[10];
rz(3*pi) q[10];
rz(4.920706033860959) q[10];
rz(1.4983086101235403) q[10];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[9];
cx q[5],q[2];
rz(-1.1847228350503194) q[2];
cx q[5],q[2];
rz(-pi/4) q[2];
cx q[5],q[6];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0.6106501184402097) q[5];
cx q[3],q[5];
rz(-0.6106501184402097) q[5];
cx q[3],q[5];
rz(pi/4) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
x q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[2],q[6];
rz(3.4801022583676025) q[6];
cx q[2],q[6];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
sx q[2];
rz(0.9892513692421632) q[6];
cx q[12],q[6];
rz(-4.109264915408066) q[6];
sx q[6];
rz(0.3459224843538591) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[12],q[6];
cx q[12],q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(0.7497198379330554) q[11];
sx q[11];
rz(6.7679752863440275) q[11];
sx q[11];
rz(14.595846299991724) q[11];
rz(pi) q[11];
rz(0.2200618061378099) q[12];
rz(5.99856687754712) q[12];
rz(0) q[6];
sx q[6];
rz(5.937262822825727) q[6];
sx q[6];
rz(12.544791506935283) q[6];
rz(-pi/2) q[6];
rz(-pi/4) q[7];
cx q[1],q[7];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(3.039715658944229) q[0];
cx q[1],q[0];
rz(-3.039715658944229) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(5.900810070219462) q[0];
sx q[0];
rz(7.881544589972925) q[0];
sx q[0];
rz(14.649816496792265) q[0];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[3];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
id q[3];
rz(0.01017352925092796) q[7];
cx q[12],q[7];
rz(-5.99856687754712) q[7];
sx q[7];
rz(2.845158079729193) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[12],q[7];
rz(pi/2) q[12];
sx q[12];
rz(6.388334997431275) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(0) q[7];
sx q[7];
rz(3.438027227450393) q[7];
sx q[7];
rz(15.413171309065572) q[7];
rz(pi/2) q[7];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[1],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
rz(1.6570834287325629) q[1];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
x q[7];
rz(2.3186882628475654) q[7];
cx q[0],q[7];
rz(-2.3186882628475654) q[7];
cx q[0],q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.6458494699096953) q[9];
cx q[4],q[9];
cx q[4],q[8];
sx q[4];
rz(1.891634537575701) q[4];
cx q[8],q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
cx q[5],q[13];
rz(-1.3604584985367698) q[13];
cx q[5],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/4) q[13];
id q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.8836595032447719) q[9];
cx q[10],q[9];
rz(-1.4983086101235403) q[9];
sx q[9];
rz(2.842521051641637) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[10],q[9];
cx q[10],q[6];
rz(-pi/4) q[10];
cx q[10],q[5];
cx q[5],q[10];
cx q[10],q[5];
rz(2.41447400654395) q[5];
cx q[10],q[5];
rz(-2.41447400654395) q[5];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(5.540783528752036) q[10];
rz(pi) q[10];
x q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[5],q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
rz(5.038118675247258) q[5];
rz(4.007805509881397) q[5];
rz(pi/2) q[6];
rz(4.188584710777293) q[6];
cx q[6],q[8];
cx q[8],q[6];
cx q[6],q[13];
cx q[13],q[6];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(5.912565204212663) q[6];
x q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.39088790758021935) q[8];
cx q[8],q[12];
rz(-0.39088790758021935) q[12];
cx q[8],q[12];
cx q[0],q[8];
rz(0.39088790758021935) q[12];
rz(4.738560635721093) q[12];
rz(4.437517692226455) q[12];
cx q[12],q[1];
rz(-4.437517692226455) q[1];
sx q[1];
rz(0.6924423867586786) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[12],q[1];
rz(0) q[1];
sx q[1];
rz(5.590742920420908) q[1];
sx q[1];
rz(12.20521222426327) q[1];
rz(-pi/2) q[12];
cx q[1],q[12];
id q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[12];
rz(0) q[12];
sx q[12];
rz(8.435279054670138) q[12];
sx q[12];
rz(3*pi) q[12];
rz(pi/2) q[12];
cx q[10],q[12];
cx q[12],q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(1.6572161375994945) q[8];
cx q[0],q[8];
rz(2.18678511744554) q[0];
cx q[8],q[0];
rz(-2.18678511744554) q[0];
cx q[8],q[0];
rz(0.13786510165823493) q[0];
rz(pi/2) q[0];
sx q[0];
rz(7.689313488490916) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.1742208845948203) q[0];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[2];
cx q[2],q[8];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.8210825114587565) q[8];
rz(-pi/4) q[8];
rz(3.28032927294717) q[8];
rz(3.2804668460069912) q[8];
cx q[8],q[0];
rz(-3.2804668460069912) q[0];
sx q[0];
rz(1.892915540571551) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[8],q[0];
rz(0) q[0];
sx q[0];
rz(4.390269766608036) q[0];
sx q[0];
rz(13.879465691371191) q[0];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(3.4406642555379494) q[9];
sx q[9];
rz(10.039427067648148) q[9];
rz(4.087910874352465) q[9];
rz(3.284849488493288) q[9];
cx q[9],q[4];
rz(-3.284849488493288) q[4];
sx q[4];
rz(2.720733212408519) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[9],q[4];
rz(0) q[4];
sx q[4];
rz(3.5624520947710674) q[4];
sx q[4];
rz(10.817992911686966) q[4];
rz(2.033217937566413) q[4];
cx q[3],q[4];
rz(-2.033217937566413) q[4];
cx q[3],q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[4];
cx q[4],q[13];
rz(-pi/4) q[13];
cx q[4],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
id q[9];
cx q[9],q[3];
cx q[3],q[9];
cx q[9],q[3];
rz(-0.8805059716503498) q[3];
sx q[3];
rz(5.482445316778162) q[3];
rz(0.22486870752729748) q[3];
cx q[5],q[3];
rz(-4.007805509881397) q[3];
sx q[3];
rz(1.3744099719042702) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[5],q[3];
rz(0) q[3];
sx q[3];
rz(4.908775335275316) q[3];
sx q[3];
rz(13.20771476312348) q[3];
rz(-pi/2) q[3];
cx q[5],q[3];
rz(pi/2) q[3];
rz(0) q[3];
sx q[3];
rz(4.71624470013703) q[3];
sx q[3];
rz(3*pi) q[3];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[2],q[5];
rz(6.279105857788231) q[5];
cx q[2],q[5];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(5.222955190609219) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[0],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(1.7964106500918873) q[5];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[11],q[9];
rz(pi/4) q[11];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[11],q[9];
rz(-pi/4) q[9];
cx q[11],q[9];
rz(pi/4) q[11];
cx q[11],q[7];
rz(-pi/4) q[7];
cx q[11],q[7];
rz(pi/2) q[11];
cx q[11],q[4];
x q[11];
cx q[4],q[3];
rz(0) q[3];
sx q[3];
rz(1.5669406070425564) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[10],q[3];
rz(pi/4) q[10];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[10],q[3];
rz(-pi/4) q[3];
cx q[10],q[3];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[12];
rz(1.9918090010801892) q[12];
cx q[10],q[12];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(1.3743511637520935) q[10];
sx q[10];
rz(6.768099214758354) q[10];
sx q[10];
rz(13.275912562640139) q[10];
rz(5.183316863790503) q[10];
rz(-pi/2) q[10];
rz(-pi/2) q[12];
rz(-6.020890142933732) q[12];
rz(pi/2) q[12];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(4.500452781209509) q[3];
rz(0.6180963179488084) q[4];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(5.237774575006031) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[11],q[7];
rz(0.8789858807057689) q[7];
cx q[11],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[13];
rz(3.7633321211994586) q[13];
cx q[9],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[6],q[13];
rz(2.8385899847760556) q[13];
cx q[6],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
cx q[4],q[13];
rz(-0.6180963179488084) q[13];
cx q[4],q[13];
rz(0.6180963179488084) q[13];
rz(-0.48485965464587144) q[13];
cx q[3],q[13];
rz(-4.500452781209509) q[13];
sx q[13];
rz(2.6547658939123138) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[3],q[13];
rz(0) q[13];
sx q[13];
rz(3.6284194132672725) q[13];
sx q[13];
rz(14.41009039662476) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(3.3586340964291987) q[3];
sx q[3];
rz(7.013042599760492) q[3];
sx q[3];
rz(15.069672283865811) q[3];
rz(pi) q[3];
x q[3];
rz(pi) q[3];
rz(1.2161129238600683) q[4];
sx q[4];
rz(6.829706042746954) q[4];
sx q[4];
rz(12.75919420958593) q[4];
rz(pi) q[4];
sx q[4];
rz(3.8326703582036816) q[4];
rz(pi/2) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
x q[6];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.9307387535782816) q[9];
rz(pi/2) q[9];
cx q[1],q[9];
rz(0) q[1];
sx q[1];
rz(1.3239101898156684) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[9];
sx q[9];
rz(1.3239101898156684) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[1],q[9];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[6],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(7.408195651788963) q[1];
sx q[1];
rz(5*pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[6];
rz(pi/2) q[6];
cx q[1],q[6];
cx q[6],q[1];
rz(4.254338710201468) q[1];
sx q[1];
rz(3.5349483506758803) q[1];
sx q[1];
rz(15.644384820641669) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
rz(-pi/2) q[9];
rz(-1.9307387535782816) q[9];
rz(0.12097017154028063) q[9];
cx q[11],q[9];
rz(-0.12097017154028063) q[9];
cx q[11],q[9];
rz(0.3702612321286791) q[11];
cx q[11],q[7];
cx q[2],q[9];
rz(-0.3702612321286791) q[7];
cx q[11],q[7];
rz(pi/2) q[11];
rz(0.3702612321286791) q[7];
cx q[7],q[8];
rz(4.1516441319215245) q[8];
cx q[7],q[8];
cx q[7],q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[12];
rz(0) q[12];
sx q[12];
rz(1.5501677191895247) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[6];
sx q[6];
rz(4.733017587990061) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[12];
rz(-pi/2) q[12];
rz(6.020890142933732) q[12];
sx q[12];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(0) q[6];
sx q[6];
rz(3.287404123134241) q[6];
sx q[6];
rz(3*pi) q[6];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
rz(0) q[4];
sx q[4];
rz(0.5667939003944364) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[7];
sx q[7];
rz(0.5667939003944364) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[4];
rz(-pi/2) q[4];
rz(-3.8326703582036816) q[4];
cx q[4],q[10];
rz(pi/2) q[10];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-2.648800066028574) q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[0];
cx q[0],q[8];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[7];
rz(0) q[0];
sx q[0];
rz(5.668857731052332) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[7];
sx q[7];
rz(0.6143275761272546) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[0],q[7];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[7];
rz(2.648800066028574) q[7];
cx q[8],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[13],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[13];
cx q[13],q[1];
rz(-pi/4) q[1];
cx q[13],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[3],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[9],q[2];
rz(pi) q[2];
x q[2];
rz(4.55108763145071) q[2];
rz(3.6861505442602875) q[2];
cx q[2],q[5];
rz(-3.6861505442602875) q[5];
sx q[5];
rz(2.362976841348882) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[2],q[5];
rz(0) q[5];
sx q[5];
rz(3.9202084658307044) q[5];
sx q[5];
rz(11.31451785493778) q[5];
rz(pi/4) q[5];
cx q[5],q[6];
rz(0) q[6];
sx q[6];
rz(2.995781184045345) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[5],q[6];
sx q[9];
cx q[11],q[9];
x q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[9];
cx q[11],q[9];
cx q[9],q[11];
rz(5.343555382610809) q[11];
sx q[11];
rz(4.7442858279112246) q[11];
sx q[11];
rz(12.677551704255965) q[11];
sx q[11];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
cx q[2],q[9];
rz(4.663489089513724) q[2];
sx q[2];
rz(8.745394580622136) q[2];
sx q[2];
rz(10.8715516456006) q[2];
rz(pi/2) q[9];
rz(5.510281770122442) q[9];
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
