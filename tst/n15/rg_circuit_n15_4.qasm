OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rz(1.6433104491694004) q[0];
sx q[0];
rz(2.891512038272955) q[0];
rz(pi/4) q[0];
rz(3.1276748682774502) q[0];
rz(-1.2574811549461853) q[0];
rz(0.6792211096862225) q[1];
rz(1.1810602848489409) q[3];
cx q[3],q[1];
rz(-1.1810602848489409) q[1];
sx q[1];
rz(0.5692721377820007) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[3],q[1];
rz(0) q[1];
sx q[1];
rz(5.7139131693975855) q[1];
sx q[1];
rz(9.926617135932098) q[1];
rz(5.902261063982591) q[1];
rz(4.272534733634246) q[1];
rz(4.624134008230767) q[3];
sx q[3];
rz(8.04664107286246) q[3];
sx q[3];
rz(10.349613758971143) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
rz(2.519327188090623) q[5];
rz(pi) q[6];
x q[6];
rz(-pi/4) q[6];
rz(pi/4) q[6];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-0.010390794399916192) q[8];
sx q[8];
rz(4.474792477815404) q[8];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[10];
cx q[10],q[7];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[11],q[5];
rz(-2.519327188090623) q[5];
cx q[11],q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[11],q[5];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
cx q[5],q[7];
cx q[7],q[5];
cx q[5],q[7];
id q[5];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[4];
cx q[4],q[12];
rz(0) q[12];
sx q[12];
rz(6.054580187648289) q[12];
sx q[12];
rz(3*pi) q[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.4941395639932058) q[4];
cx q[1],q[4];
rz(-4.272534733634246) q[4];
sx q[4];
rz(1.5622126055408285) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[1],q[4];
rz(0) q[4];
sx q[4];
rz(4.7209727016387575) q[4];
sx q[4];
rz(12.20317313041042) q[4];
rz(pi/4) q[4];
cx q[4],q[10];
rz(-pi/4) q[10];
cx q[4],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-0.23406001771829676) q[10];
sx q[10];
rz(4.840486937890127) q[10];
sx q[10];
rz(9.658837978487677) q[10];
rz(2.0488006388539484) q[10];
cx q[5],q[10];
rz(-2.0488006388539484) q[10];
cx q[5],q[10];
rz(-pi/2) q[10];
rz(pi/4) q[5];
cx q[8],q[12];
rz(0) q[12];
sx q[12];
rz(0.2286051195312968) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[8],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[6],q[12];
rz(-pi/4) q[12];
cx q[6],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(3.0773614899813992) q[12];
rz(pi/2) q[12];
rz(3.325321423640327) q[6];
cx q[6],q[0];
rz(-3.325321423640327) q[0];
sx q[0];
rz(2.6332870596571647) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[6],q[0];
rz(0) q[0];
sx q[0];
rz(3.6498982475224215) q[0];
sx q[0];
rz(14.007580539355892) q[0];
rz(4.90853460730545) q[0];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.3908276212430246) q[6];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[9],q[13];
rz(5.54490840106418) q[13];
cx q[9],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi) q[13];
rz(-pi/2) q[13];
cx q[13],q[7];
rz(1.7230481596074492) q[7];
cx q[13],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-0.35094915378897973) q[9];
sx q[9];
rz(7.585058935129718) q[9];
cx q[9],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
cx q[3],q[11];
cx q[3],q[11];
rz(0.15393800587295292) q[11];
cx q[3],q[11];
rz(-0.7334069427078443) q[11];
sx q[11];
rz(5.024596172592416) q[11];
rz(2.5376839770142774) q[3];
cx q[11],q[3];
cx q[3],q[11];
cx q[11],q[3];
rz(pi/2) q[11];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[11];
cx q[11],q[3];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[3];
rz(1.8926085546716727) q[3];
sx q[3];
rz(-3.634901765491218) q[9];
sx q[9];
rz(8.772761083411835) q[9];
sx q[9];
rz(13.059679726260597) q[9];
cx q[9],q[13];
cx q[13],q[9];
rz(3.5828081666282396) q[13];
cx q[9],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[9];
cx q[9],q[10];
id q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
cx q[10],q[3];
x q[10];
sx q[10];
rz(1.6554837063938457) q[3];
sx q[3];
rz(8.830647692335884) q[3];
sx q[3];
rz(13.317855681200825) q[3];
rz(0.8782228922996371) q[3];
x q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[2],q[14];
cx q[14],q[2];
cx q[14],q[2];
rz(0.6444578442115506) q[2];
cx q[14],q[2];
cx q[14],q[1];
cx q[1],q[14];
cx q[14],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[14],q[4];
rz(0) q[2];
sx q[2];
rz(4.351595224781802) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[4],q[14];
cx q[14],q[4];
rz(0.5946644755057076) q[14];
cx q[0],q[14];
rz(-4.90853460730545) q[14];
sx q[14];
rz(2.476071261118892) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[0],q[14];
rz(3.1820518418499892) q[0];
sx q[0];
rz(1.7135513641154392) q[0];
rz(0.2518820236018931) q[0];
sx q[0];
rz(8.915965921607011) q[0];
sx q[0];
rz(15.632403315025696) q[0];
cx q[0],q[9];
rz(0) q[14];
sx q[14];
rz(3.8071140460606943) q[14];
sx q[14];
rz(13.738648092569122) q[14];
rz(4.452072910646863) q[14];
cx q[4],q[6];
rz(-0.3908276212430246) q[6];
cx q[4],q[6];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[5],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.556099646319619) q[4];
cx q[4],q[13];
rz(-2.556099646319619) q[13];
cx q[4],q[13];
rz(2.556099646319619) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3.207654010644836) q[6];
rz(pi/2) q[6];
cx q[8],q[2];
rz(0) q[2];
sx q[2];
rz(1.9315900823977843) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[8],q[2];
rz(0.5209155156246909) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(3.0811563881880093) q[2];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[1];
rz(1.7995928184431806) q[1];
cx q[8],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[7];
rz(pi/4) q[1];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[1],q[7];
rz(-pi/4) q[7];
cx q[1],q[7];
rz(1.4491590899771487) q[1];
cx q[2],q[1];
rz(-3.0811563881880093) q[1];
sx q[1];
rz(0.9007782877870927) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[2],q[1];
rz(0) q[1];
sx q[1];
rz(5.3824070193924936) q[1];
sx q[1];
rz(11.05677525898024) q[1];
sx q[1];
rz(pi/4) q[2];
cx q[2],q[5];
rz(-pi/4) q[5];
cx q[2],q[5];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[13];
rz(1.0083993427172808) q[13];
cx q[2],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-1.399401468644654) q[5];
sx q[5];
rz(7.69383907678112) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[1];
x q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[4];
rz(0.29085505802077505) q[4];
cx q[6],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[5],q[6];
rz(4.541529764174159) q[6];
cx q[5],q[6];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[5],q[6];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(3.815173669746042) q[6];
sx q[6];
rz(4.462188955682821) q[6];
sx q[6];
rz(14.869999475265026) q[6];
rz(2.241795427793783) q[6];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[7];
cx q[12],q[7];
x q[12];
rz(0) q[12];
sx q[12];
rz(5.582404146082799) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[7],q[12];
rz(0) q[12];
sx q[12];
rz(0.7007811610967876) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[7],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[1],q[12];
rz(pi/4) q[1];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[1],q[12];
rz(-pi/4) q[12];
cx q[1],q[12];
rz(pi/4) q[1];
cx q[1],q[11];
rz(-pi/4) q[11];
cx q[1],q[11];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[4],q[12];
rz(4.0147337557101705) q[12];
cx q[4],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.7492850602739725) q[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
x q[7];
rz(0.5629817943034089) q[7];
x q[7];
rz(pi/4) q[7];
cx q[7],q[1];
rz(-pi/4) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[1];
rz(1.735151331213733) q[1];
rz(5.656461883591752) q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
cx q[14],q[8];
rz(3.22861675415305) q[14];
sx q[14];
rz(5.7908845601748276) q[14];
sx q[14];
rz(14.308231311245951) q[14];
rz(-1.9767619887161285) q[14];
rz(2.747685266719446) q[8];
rz(pi/4) q[8];
cx q[8],q[13];
rz(-pi/4) q[13];
cx q[8],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[11],q[13];
cx q[13],q[11];
cx q[11],q[13];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[13];
cx q[5],q[13];
cx q[13],q[5];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(4.408166629897367) q[8];
rz(3.6247681742346076) q[9];
cx q[0],q[9];
rz(3.549856682941459) q[0];
cx q[0],q[14];
rz(-3.549856682941459) q[14];
sx q[14];
rz(0.38100376100037847) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[0],q[14];
rz(0.3977528909265917) q[0];
rz(0) q[14];
sx q[14];
rz(5.902181546179207) q[14];
sx q[14];
rz(14.951396632426967) q[14];
sx q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[8],q[0];
rz(-4.408166629897367) q[0];
sx q[0];
rz(2.523593230442324) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[8],q[0];
rz(0) q[0];
sx q[0];
rz(3.7595920767372624) q[0];
sx q[0];
rz(13.435191699740155) q[0];
rz(3.0475425248093653) q[0];
rz(pi/2) q[0];
cx q[11],q[0];
rz(0) q[0];
sx q[0];
rz(2.258591024846917) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[11];
sx q[11];
rz(2.258591024846917) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[11],q[0];
rz(-pi/2) q[0];
rz(-3.0475425248093653) q[0];
cx q[0],q[13];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(2.7516118119373614) q[11];
rz(3.855755096825811) q[13];
cx q[0],q[13];
rz(-pi/4) q[0];
rz(4.796429012711768) q[0];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(-pi/4) q[8];
rz(-0.8745899644927488) q[8];
cx q[11],q[8];
rz(-2.7516118119373614) q[8];
sx q[8];
rz(2.68984211425165) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[11],q[8];
rz(3.250434638526891) q[11];
cx q[11],q[3];
rz(-3.250434638526891) q[3];
sx q[3];
rz(1.8864153603230407) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[11],q[3];
sx q[11];
rz(1.0999420633157488) q[11];
rz(pi/2) q[11];
rz(0) q[3];
sx q[3];
rz(4.396769946856546) q[3];
sx q[3];
rz(11.796989706996634) q[3];
rz(0) q[8];
sx q[8];
rz(3.5933431929279362) q[8];
sx q[8];
rz(13.05097973719949) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[2],q[9];
cx q[9],q[2];
cx q[2],q[9];
cx q[12],q[2];
rz(-2.7492850602739725) q[2];
cx q[12],q[2];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0.5874007033851021) q[12];
rz(2.7492850602739725) q[2];
rz(pi) q[2];
x q[2];
cx q[4],q[12];
rz(-0.5874007033851021) q[12];
cx q[4],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(3.1219233514879274) q[12];
cx q[4],q[14];
rz(-pi/2) q[14];
cx q[4],q[14];
rz(pi/2) q[14];
rz(0.1497560721578397) q[14];
cx q[0],q[14];
rz(-4.796429012711768) q[14];
sx q[14];
rz(0.0042788650394829375) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[0],q[14];
rz(0) q[14];
sx q[14];
rz(6.278906442140103) q[14];
sx q[14];
rz(14.071450901323308) q[14];
rz(2.1837878602800536) q[14];
rz(1.2421065770300785) q[4];
rz(-0.26078689062645277) q[4];
cx q[5],q[12];
rz(-3.1219233514879274) q[12];
cx q[5],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
x q[13];
rz(4.257963467573354) q[13];
cx q[13],q[4];
rz(-4.257963467573354) q[4];
sx q[4];
rz(1.1573496182140413) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[13],q[4];
rz(0.16733043231490496) q[13];
sx q[13];
rz(4.767991122127466) q[13];
rz(0.10495791114077274) q[13];
rz(3.3292449734777563) q[13];
rz(0) q[4];
sx q[4];
rz(5.1258356889655445) q[4];
sx q[4];
rz(13.943528318969186) q[4];
rz(3.9839572912882866) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rz(-5.08133864348685) q[5];
rz(pi/2) q[5];
cx q[6],q[2];
rz(-2.241795427793783) q[2];
cx q[6],q[2];
rz(2.241795427793783) q[2];
rz(3.6004821476310855) q[2];
rz(0) q[6];
sx q[6];
rz(5.991829083430693) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[10],q[6];
rz(0) q[6];
sx q[6];
rz(0.29135622374889314) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[10],q[6];
cx q[1],q[10];
rz(5.651314923627137) q[10];
cx q[1],q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[11];
rz(0) q[10];
sx q[10];
rz(0.41157827221489995) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[11];
sx q[11];
rz(0.41157827221489995) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[10],q[11];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi) q[10];
x q[10];
rz(pi/2) q[10];
sx q[10];
rz(8.57687140735669) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(-pi/2) q[11];
rz(-1.0999420633157488) q[11];
rz(pi/2) q[11];
rz(-0.977009158596801) q[6];
sx q[6];
rz(7.151666896578771) q[6];
rz(4.304685306520223) q[9];
rz(4.63103377228567) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[7];
rz(0) q[7];
sx q[7];
rz(0.218341695796751) q[7];
sx q[7];
rz(3*pi) q[7];
rz(0) q[9];
sx q[9];
rz(0.218341695796751) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[7];
rz(-pi/2) q[7];
rz(-5.656461883591752) q[7];
rz(-0.4321747706767878) q[7];
cx q[2],q[7];
rz(-3.6004821476310855) q[7];
sx q[7];
rz(0.022714213588515886) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[2],q[7];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[5];
rz(0) q[2];
sx q[2];
rz(4.470167234666349) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[5];
sx q[5];
rz(1.813018072513238) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[2],q[5];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[5];
rz(5.08133864348685) q[5];
cx q[5],q[6];
rz(0.19798710185196078) q[6];
cx q[5],q[6];
rz(-pi/4) q[5];
rz(1.268525321398133) q[5];
sx q[5];
rz(2.7596285972288417) q[5];
rz(2.8833813372724517) q[5];
rz(pi/2) q[6];
sx q[6];
rz(4.5035492147045) q[6];
sx q[6];
rz(5*pi/2) q[6];
rz(0) q[7];
sx q[7];
rz(6.26047109359107) q[7];
sx q[7];
rz(13.457434879077253) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[12],q[7];
rz(5.940633531507581) q[7];
cx q[12],q[7];
id q[12];
rz(pi/2) q[12];
rz(-0.7834554808356071) q[12];
cx q[13],q[12];
rz(-3.3292449734777563) q[12];
sx q[12];
rz(1.3058234725985682) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[13],q[12];
rz(0) q[12];
sx q[12];
rz(4.977361834581018) q[12];
sx q[12];
rz(13.537478415082742) q[12];
rz(pi/4) q[13];
rz(0) q[13];
sx q[13];
rz(6.269291755914995) q[13];
sx q[13];
rz(3*pi) q[13];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[1],q[7];
rz(pi/4) q[1];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[1],q[7];
rz(-pi/4) q[7];
cx q[1],q[7];
rz(2.4195618406136563) q[1];
sx q[1];
cx q[5],q[1];
rz(-2.8833813372724517) q[1];
cx q[5],q[1];
rz(2.8833813372724517) q[1];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[1],q[5];
rz(pi/4) q[1];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[1],q[5];
rz(-pi/4) q[5];
cx q[1],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.7494966567544564) q[7];
sx q[7];
rz(4.477327946776279) q[7];
sx q[7];
rz(13.61985882617363) q[7];
rz(-pi/2) q[7];
cx q[6],q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[10],q[7];
rz(pi/2) q[10];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(pi/4) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(-pi/4) q[9];
cx q[8],q[9];
cx q[3],q[8];
cx q[8],q[3];
cx q[2],q[8];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[8],q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[11];
cx q[11],q[2];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0.08987190944837332) q[8];
sx q[8];
rz(3.739488881331585) q[8];
sx q[8];
rz(11.034558995256457) q[8];
rz(pi/4) q[8];
rz(4.710963457604806) q[8];
sx q[8];
rz(3.8967408600755244) q[8];
sx q[8];
rz(11.037421557284533) q[8];
rz(2.5069125592937325) q[8];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(2.648721982082251) q[9];
sx q[9];
rz(7.791241213980312) q[9];
sx q[9];
rz(10.687338156337983) q[9];
cx q[0],q[9];
rz(3.8247402323128052) q[9];
cx q[0],q[9];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[4];
rz(0) q[0];
sx q[0];
rz(0.9451252306045266) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[14],q[9];
rz(0) q[4];
sx q[4];
rz(0.9451252306045266) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[0],q[4];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[11],q[0];
rz(0.033315693137947924) q[0];
cx q[11],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[0];
x q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.8522235326296013) q[0];
cx q[11],q[6];
rz(-pi/2) q[4];
rz(-3.9839572912882866) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(4.018933278162176) q[6];
cx q[11],q[6];
cx q[11],q[0];
rz(-2.8522235326296013) q[0];
cx q[11],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.7147067768147186) q[0];
sx q[0];
rz(3.819546223734338) q[0];
sx q[0];
rz(11.593576012490075) q[0];
rz(pi) q[11];
rz(3.021467583441718) q[6];
sx q[6];
rz(5.606485342639806) q[6];
sx q[6];
rz(14.489729603077702) q[6];
id q[6];
rz(3.5824542267488866) q[9];
cx q[14],q[9];
rz(1.6993173760045321) q[14];
cx q[14],q[2];
rz(-1.6993173760045321) q[2];
cx q[14],q[2];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.2985792817602604) q[14];
rz(1.6993173760045321) q[2];
rz(2.9055843582488654) q[2];
cx q[2],q[4];
rz(-2.9055843582488654) q[4];
cx q[2],q[4];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.9055843582488654) q[4];
rz(0.5348666081276163) q[4];
cx q[10],q[4];
cx q[4],q[10];
rz(pi/4) q[9];
cx q[9],q[3];
rz(-pi/4) q[3];
cx q[9],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[12];
cx q[12],q[3];
cx q[3],q[12];
cx q[12],q[13];
rz(0) q[13];
sx q[13];
rz(0.013893551264591242) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[12],q[13];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[5];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(3.837829782328698) q[3];
cx q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[5],q[12];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[8],q[3];
rz(0.34908187348868325) q[3];
cx q[8],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[9],q[14];
rz(-0.2985792817602604) q[14];
cx q[9],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(3.2186913500470404) q[14];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[1];
cx q[1],q[7];
cx q[7],q[1];
rz(pi) q[9];
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