// I have added the missing zones for the upper range
// The idea behind this indicator: https://www.lookintobitcoin.com/charts/bitcoin-logarithmic-growth-curve/

//@version=4
// This code is based on https://www.tradingview.com/script/jX8rqCHD-Bitcoin-Logarithmic-Growth-Curves-for-intraday-users/
// I've left all the original comments.
// © krzyczak
// Lower zones are not commented out, as trading view does not cope with the amount of data.
// It needs to be split into two indicators (lower zones and upper zones) to be able to display all zones.
//
// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © aaahopper
//
// The indicatotor was written by @Quantadelic aka Pinescript Pleb aka Stealer of Alpha & Destroyer of Discords.
// I have left his statements :)
// "I drink your milkshake! I drink it up."
// No rights reserved, copy, paste, do what you want.

study(title="[All Zones] Bitcoin Logarithmic Growth Curves for intraday users", shorttitle="[All Zones] Modified Bitcoin Log Growth Curves for intraday users", overlay=true)

// INPUTS
only =          input(false, title="Show all period")
Hoursback =     input(700, title='Hours to review')
HighIntercept = input(1.06930947, title="Top Curve Intercept", step=0.001)
HighSlope =     input(0.00076, title="Top Curve Slope", step=0.00001)
LowIntercept =  input(-3.0269716, title="Bottom Curve Intercept", step=0.001)
LowSlope =      input(0.001329, title="Bottom Curve Slope", step=0.00001)
Zone10 = input(true, title='Display Zone 10')
Zone09 = input(true, title='Display Zone 9')
Zone08 = input(true, title='Display Zone 8')
Zone07 = input(true, title='Display Zone 7')
Zone06 = input(true, title='Display Zone 6')
Zone05 = input(true, title='Display Zone 5 (MidLine)')
Zone04 = input(false, title='Display Zone 4')
Zone03 = input(false, title='Display Zone 3')
Zone02 = input(false, title='Display Zone 2')
Zone01 = input(false, title='Display Zone 1 (Lowest)')


//Length of time required to preview =========================
yo = Hoursback - hour
oneday = 1000 * 60 * 60 * yo
clean = only ? true : timenow - time < oneday

//Calculating High and Low ranges =========================
// TIME CALCS
TimeIndex =     time < 1279670400000 ? 3.0 : (time - 1279670400000) / 86400000
Weight =        (log10(TimeIndex + 10) * TimeIndex * TimeIndex - TimeIndex) / 30000
TimeDelta =     time - time[1]
// TOP OF CHANNEL INTERCEPT & SLOPE CALCS
HighSlopeCum =  HighSlope * TimeIndex
HighLogDev =    TimeIndex > 2 ? log(Weight) + HighIntercept + HighSlopeCum : na
// BOTTOM OF CHANNEL INTERCEPT & SLOPE CALCS
LowSlopeCum =   LowSlope * TimeIndex
LowLogDev =     TimeIndex > 2 ? log(Weight) + LowIntercept + LowSlopeCum : na
// TOTAL CHANNEL LOG RANGE
LogRange =      HighLogDev - LowLogDev

// FIBONACCI LEVELS
Fib9098Calc =   (LogRange * 0.9098) + LowLogDev
Fib8541Calc =   (LogRange * 0.8541) + LowLogDev
Fib7639Calc =   (LogRange * 0.7639) + LowLogDev
Fib618Calc =    (LogRange * 0.618) + LowLogDev
MidCalc =       (LogRange * 0.5) + LowLogDev
Fib382Calc =    (LogRange * 0.382) + LowLogDev
Fib2361Calc =   (LogRange * 0.2361) + LowLogDev
Fib1459Calc =   (LogRange * 0.1459) + LowLogDev
Fib0902Calc =   (LogRange * 0.0902) + LowLogDev

// SCALE LOG VARIABLES TO LINEAR
HighDev =       pow(2.718281828459, HighLogDev)
Fib9098Dev =    pow(2.718281828459, Fib9098Calc)
Fib8541Dev =    pow(2.718281828459, Fib8541Calc)
Fib7639Dev =    pow(2.718281828459, Fib7639Calc)
Fib618Dev =     pow(2.718281828459, Fib618Calc)
MidDev =        pow(2.718281828459, MidCalc)
Fib382Dev =      pow(2.718281828459, Fib382Calc)
Fib2361Dev =    pow(2.718281828459, Fib2361Calc)
Fib1459Dev =    pow(2.718281828459, Fib1459Calc)
Fib0902Dev =    pow(2.718281828459, Fib0902Calc)
LowDev =           pow(2.718281828459, LowLogDev)


// PLOTS
L10_00 = plot(clean and Zone10?HighDev:na, color=color.black, transp=0, title="100% - Top of Channel")
L9_00 = plot(clean and Zone09?Fib9098Dev:na, color=color.black, transp=50, title="90.98% - Fibonacci Level")
L8_00 = plot(clean and Zone08?Fib8541Dev:na, color=color.black, transp=50, title="85.41% - Fibonacci Level")
L7_00 = plot(clean and Zone07?Fib7639Dev:na, color=color.black, transp=50, title="76.39% - Fibonacci Level")
L6_00 = plot(clean and Zone06?Fib618Dev:na, color=color.black, transp=0, title="61.80% - Fibonacci Level",linewidth=2)
L5_00 = plot(clean and Zone05?MidDev:na, color=color.white, transp=0, title="50% - Middle of Channel",linewidth=2)
L4_00 = plot(clean and (Zone04 or Zone05)?Fib382Dev:na, color=color.black, transp=0, title="38.20% - Fibonacci Level",linewidth=2)
L3_00 = plot(clean and (Zone03 or Zone04)?Fib2361Dev:na, color=color.black, transp=0, title="23.61% - Fibonacci Level",linewidth=2)
L2_00 = plot(clean and (Zone02 or Zone03)?Fib1459Dev:na, color=color.black, transp=0, title="14.59% - Fibonacci Level",linewidth=2)
L1_00 = plot(clean and (Zone01 or Zone02)?Fib0902Dev:na, color=color.black, transp=0, title="9.02% - Fibonacci Level",linewidth=2)
L0_00 = plot(clean and Zone01?LowDev:na, color=color.black, transp=0, title="0% - Bottom of Channel",linewidth=2)

// // Level - from LowDev - Fib0902Dev ====================================
// ranr1 = Fib0902Dev - LowDev
// LowDev0 = Fib0902Dev - 0.1100 * ranr1
// LowDev1 = Fib0902Dev - 0.236 * ranr1
// LowDev2 = Fib0902Dev - 0.382 * ranr1
// LowDev3 = Fib0902Dev - 0.500 * ranr1
// LowDev4 = LowDev + 0.382 * ranr1
// LowDev5 = LowDev + 0.236 * ranr1
// LowDev6 = LowDev + 0.1100 * ranr1

// L1_01 = plot(clean and Zone01?LowDev0:na, color=color.red, transp=0, title="0% - Bottom of Channel")
// L1_11 = plot(clean and Zone01?LowDev1:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L1_21 = plot(clean and Zone01?LowDev2:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L1_31 = plot(clean and Zone01?LowDev3:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L1_41 = plot(clean and Zone01?LowDev4:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L1_51 = plot(clean and Zone01?LowDev5:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L1_61 = plot(clean and Zone01?LowDev6:na, color=color.green, transp=0, title="0% - Bottom of Channel")
// fill(L1_00,L1_01,color=color.fuchsia,transp=50)
// fill(L0_00,L1_61,color=color.lime,transp=50)


// // Level 2 - from Fib0902Dev - Fib1459Dev ====================================
// ranr12 = Fib1459Dev - Fib0902Dev
// LowDev02 = Fib1459Dev - 0.1100 * ranr12
// LowDev12 = Fib1459Dev - 0.236 * ranr12
// LowDev22 = Fib1459Dev - 0.382 * ranr12
// LowDev32 = Fib1459Dev - 0.500 * ranr12
// LowDev42 = Fib0902Dev + 0.382 * ranr12
// LowDev52 = Fib0902Dev + 0.236 * ranr12
// LowDev62 = Fib0902Dev + 0.1100 * ranr12

// L2_02 = plot(clean and Zone02?LowDev02:na, color=color.red, transp=0, title="0% - Bottom of Channel")
// L2_12 = plot(clean and Zone02?LowDev12:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L2_22 = plot(clean and Zone02?LowDev22:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L2_32 = plot(clean and Zone02?LowDev32:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L2_42 = plot(clean and Zone02?LowDev42:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L2_52 = plot(clean and Zone02?LowDev52:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L2_62 = plot(clean and Zone02?LowDev62:na, color=color.green, transp=0, title="0% - Bottom of Channel")
// fill(L2_00,L2_02,color=color.fuchsia,transp=50)
// fill(L1_00,L2_62,color=color.lime,transp=50)


// // Level 3 - from Fib1459Dev - Fib2361Dev ====================================
// ranr13 = Fib2361Dev - Fib1459Dev
// LowDev03 = Fib2361Dev - 0.1100 * ranr13
// LowDev13 = Fib2361Dev - 0.236 * ranr13
// LowDev23 = Fib2361Dev - 0.382 * ranr13
// LowDev33 = Fib2361Dev - 0.500 * ranr13
// LowDev43 = Fib1459Dev + 0.382 * ranr13
// LowDev53 = Fib1459Dev + 0.236 * ranr13
// LowDev63 = Fib1459Dev + 0.1100 * ranr13

// L3_03 = plot(clean and Zone03?LowDev03:na, color=color.red, transp=0, title="0% - Bottom of Channel")
// L3_13 = plot(clean and Zone03?LowDev13:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L3_23 = plot(clean and Zone03?LowDev23:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L3_33 = plot(clean and Zone03?LowDev33:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L3_43 = plot(clean and Zone03?LowDev43:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L3_53 = plot(clean and Zone03?LowDev53:na, color=color.black, transp=0, title="0% - Bottom of Channel")
// L3_63 = plot(clean and Zone03?LowDev63:na, color=color.green, transp=0, title="0% - Bottom of Channel")
// fill(L3_00,L3_03,color=color.fuchsia,transp=50)
// fill(L2_00,L3_63,color=color.lime,transp=50)

// Level 4 - from Fib2361Dev - Fib382Dev ====================================
ranr14 = Fib382Dev - Fib2361Dev
LowDev04 = Fib382Dev - 0.1100 * ranr14
LowDev14 = Fib382Dev - 0.236 * ranr14
LowDev24 = Fib382Dev - 0.382 * ranr14
LowDev34 = Fib382Dev - 0.500 * ranr14
LowDev44 = Fib2361Dev + 0.382 * ranr14
LowDev54 = Fib2361Dev + 0.236 * ranr14
LowDev64 = Fib2361Dev + 0.1100 * ranr14

L4_04 = plot(clean and Zone04?LowDev04:na, color=color.red, transp=0, title="0% - Bottom of Channel")
L4_14 = plot(clean and Zone04?LowDev14:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L4_24 = plot(clean and Zone04?LowDev24:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L4_34 = plot(clean and Zone04?LowDev34:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L4_44 = plot(clean and Zone04?LowDev44:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L4_54 = plot(clean and Zone04?LowDev54:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L4_64 = plot(clean and Zone04?LowDev64:na, color=color.green, transp=0, title="0% - Bottom of Channel")
fill(L4_00,L4_04,color=color.fuchsia,transp=50)
fill(L3_00,L4_64,color=color.lime,transp=50)

// Level 5 - from Fib382Dev - MidDev ====================================
ranr15 = MidDev - Fib382Dev
LowDev05 = MidDev - 0.1100 * ranr15
LowDev15 = MidDev - 0.236 * ranr15
LowDev25 = MidDev - 0.382 * ranr15
LowDev35 = MidDev - 0.500 * ranr15
LowDev45 = Fib382Dev + 0.382 * ranr15
LowDev55 = Fib382Dev + 0.236 * ranr15
LowDev65 = Fib382Dev + 0.1100 * ranr15

L5_05 = plot(clean and Zone05?LowDev05:na, color=color.red, transp=0, title="0% - Bottom of Channel")
L5_15 = plot(clean and Zone05?LowDev15:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L5_25 = plot(clean and Zone05?LowDev25:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L5_35 = plot(clean and Zone05?LowDev35:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L5_45 = plot(clean and Zone05?LowDev45:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L5_55 = plot(clean and Zone05?LowDev55:na, color=color.black, transp=0, title="0% - Bottom of Channel")
L5_65 = plot(clean and Zone05?LowDev65:na, color=color.green, transp=0, title="0% - Bottom of Channel")
fill(L5_00,L5_05,color=color.fuchsia,transp=50)
fill(L4_00,L5_65,color=color.lime,transp=50)


// Level 6 - from MidDev - Fib618Dev ====================================
ranr16 = Fib618Dev - MidDev
LowDev06 = Fib618Dev - 0.1100 * ranr16
LowDev16 = Fib618Dev - 0.236 * ranr16
LowDev26 = Fib618Dev - 0.382 * ranr16
LowDev36 = Fib618Dev - 0.500 * ranr16
LowDev46 = MidDev + 0.382 * ranr16
LowDev56 = MidDev + 0.236 * ranr16
LowDev66 = MidDev + 0.1100 * ranr16

L6_06 = plot(clean and Zone06?LowDev06:na, color=color.red, transp=0, title="61.80% - Fibonacci Level")
L6_16 = plot(clean and Zone06?LowDev16:na, color=color.black, transp=0, title="61.80% - Fibonacci Level")
L6_26 = plot(clean and Zone06?LowDev26:na, color=color.black, transp=0, title="61.80% - Fibonacci Level")
L6_36 = plot(clean and Zone06?LowDev36:na, color=color.black, transp=0, title="61.80% - Fibonacci Level")
L6_46 = plot(clean and Zone06?LowDev46:na, color=color.black, transp=0, title="61.80% - Fibonacci Level")
L6_56 = plot(clean and Zone06?LowDev56:na, color=color.black, transp=0, title="61.80% - Fibonacci Level")
L6_66 = plot(clean and Zone06?LowDev66:na, color=color.green, transp=0, title="61.80% - Fibonacci Level")

fill(L6_06,L6_00,color=color.fuchsia,transp=50)
fill(L5_00,L6_66,color=color.lime,transp=50)


// Level 7 - from Fib618Dev - Fib7639Dev ====================================
ranr17 = Fib7639Dev - Fib618Dev
LowDev07 = Fib7639Dev - 0.1100 * ranr17
LowDev17 = Fib7639Dev - 0.236 * ranr17
LowDev27 = Fib7639Dev - 0.382 * ranr17
LowDev37 = Fib7639Dev - 0.500 * ranr17
LowDev47 = Fib618Dev + 0.382 * ranr17
LowDev57 = Fib618Dev + 0.236 * ranr17
LowDev67 = Fib618Dev + 0.1100 * ranr17

L7_07 = plot(clean and Zone07?LowDev07:na, color=color.red, transp=0, title="76.39% - Fibonacci Level")
L7_17 = plot(clean and Zone07?LowDev17:na, color=color.black, transp=0, title="76.39% - Fibonacci Level")
L7_27 = plot(clean and Zone07?LowDev27:na, color=color.black, transp=0, title="76.39% - Fibonacci Level")
L7_37 = plot(clean and Zone07?LowDev37:na, color=color.black, transp=0, title="76.39% - Fibonacci Level")
L7_47 = plot(clean and Zone07?LowDev47:na, color=color.black, transp=0, title="76.39% - Fibonacci Level")
L7_57 = plot(clean and Zone07?LowDev57:na, color=color.black, transp=0, title="76.39% - Fibonacci Level")
L7_67 = plot(clean and Zone07?LowDev67:na, color=color.green, transp=0, title="76.39% - Fibonacci Level")

fill(L7_07,L7_00,color=color.fuchsia,transp=50)
fill(L6_00,L7_67,color=color.lime,transp=50)


// Level 8 - from Fib7639Dev - Fib8541Dev ====================================
ranr18 = Fib8541Dev - Fib7639Dev
LowDev08 = Fib8541Dev - 0.1100 * ranr18
LowDev18 = Fib8541Dev - 0.236 * ranr18
LowDev28 = Fib8541Dev - 0.382 * ranr18
LowDev38 = Fib8541Dev - 0.500 * ranr18
LowDev48 = Fib7639Dev + 0.382 * ranr18
LowDev58 = Fib7639Dev + 0.236 * ranr18
LowDev68 = Fib7639Dev + 0.1100 * ranr18

L8_08 = plot(clean and Zone08?LowDev08:na, color=color.red, transp=0, title="85.41% - Fibonacci Level")
L8_18 = plot(clean and Zone08?LowDev18:na, color=color.black, transp=0, title="85.41% - Fibonacci Level")
L8_28 = plot(clean and Zone08?LowDev28:na, color=color.black, transp=0, title="85.41% - Fibonacci Level")
L8_38 = plot(clean and Zone08?LowDev38:na, color=color.black, transp=0, title="85.41% - Fibonacci Level")
L8_48 = plot(clean and Zone08?LowDev48:na, color=color.black, transp=0, title="85.41% - Fibonacci Level")
L8_58 = plot(clean and Zone08?LowDev58:na, color=color.black, transp=0, title="85.41% - Fibonacci Level")
L8_68 = plot(clean and Zone08?LowDev68:na, color=color.green, transp=0, title="85.41% - Fibonacci Level")

fill(L8_08,L8_00,color=color.fuchsia,transp=50)
fill(L7_00,L8_68,color=color.lime,transp=50)


// Level 9 - from Fib8541Dev - Fib9098Dev ====================================
ranr19 = Fib9098Dev - Fib8541Dev
LowDev09 = Fib9098Dev - 0.1100 * ranr19
LowDev19 = Fib9098Dev - 0.236 * ranr19
LowDev29 = Fib9098Dev - 0.382 * ranr19
LowDev39 = Fib9098Dev - 0.500 * ranr19
LowDev49 = Fib8541Dev + 0.382 * ranr19
LowDev59 = Fib8541Dev + 0.236 * ranr19
LowDev69 = Fib8541Dev + 0.1100 * ranr19

L9_09 = plot(clean and Zone09?LowDev09:na, color=color.red, transp=0, title="90.98% - Fibonacci Level")
L9_19 = plot(clean and Zone09?LowDev19:na, color=color.black, transp=0, title="90.98% - Fibonacci Level")
L9_29 = plot(clean and Zone09?LowDev29:na, color=color.black, transp=0, title="90.98% - Fibonacci Level")
L9_39 = plot(clean and Zone09?LowDev39:na, color=color.black, transp=0, title="90.98% - Fibonacci Level")
L9_49 = plot(clean and Zone09?LowDev49:na, color=color.black, transp=0, title="90.98% - Fibonacci Level")
L9_59 = plot(clean and Zone09?LowDev59:na, color=color.black, transp=0, title="90.98% - Fibonacci Level")
L9_69 = plot(clean and Zone09?LowDev69:na, color=color.green, transp=0, title="90.98% - Fibonacci Level")

fill(L9_09,L9_00,color=color.fuchsia,transp=50)
fill(L8_00,L9_69,color=color.lime,transp=50)


// Level 10 - from Fib9098Dev - HighDev ====================================
ranr20 = HighDev - Fib9098Dev
LowDev010 = HighDev - 0.1100 * ranr20
LowDev110 = HighDev - 0.236 * ranr20
LowDev210 = HighDev - 0.382 * ranr20
LowDev310 = HighDev - 0.500 * ranr20
LowDev410 = Fib9098Dev + 0.382 * ranr20
LowDev510 = Fib9098Dev + 0.236 * ranr20
LowDev610 = Fib9098Dev + 0.1100 * ranr20

L10_010 = plot(clean and Zone10?LowDev010:na, color=color.red, transp=0, title="100.00% - Fibonacci Level")
L10_110 = plot(clean and Zone10?LowDev110:na, color=color.black, transp=0, title="100.00% - Fibonacci Level")
L10_210 = plot(clean and Zone10?LowDev210:na, color=color.black, transp=0, title="100.00% - Fibonacci Level")
L10_310 = plot(clean and Zone10?LowDev310:na, color=color.black, transp=0, title="100.00% - Fibonacci Level")
L10_410 = plot(clean and Zone10?LowDev410:na, color=color.black, transp=0, title="100.00% - Fibonacci Level")
L10_510 = plot(clean and Zone10?LowDev510:na, color=color.black, transp=0, title="100.00% - Fibonacci Level")
L10_610 = plot(clean and Zone10?LowDev610:na, color=color.green, transp=0, title="100.00% - Fibonacci Level")

fill(L10_010,L10_00,color=color.fuchsia,transp=50)
fill(L9_00,L10_610,color=color.lime,transp=50)
