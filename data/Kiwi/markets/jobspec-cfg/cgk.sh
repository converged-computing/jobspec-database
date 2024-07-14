#!/bin/bash
# cgk.sh -- coingecko.com api access
# v0.14  feb/2021  by mountaineerbr

#defaults

#default crypto, defaults=btc
DEFCUR=btc

#vs currency, defaults=usd
DEFVSCUR=usd

#scale, defaults=16
SCLDEFAULTS=16

#don't change these
#set number format
#export LC_NUMERIC=C
export LC_NUMERIC=en_US.UTF-8

#troy ounce to gram ratio
TOZ=31.1034768

#script name
SN="${0##*/}"

## Manual and help
HELP_LINES="NAME
	Cgk.sh -- Currency Converter and Market Stats
		  Coingecko.com API Access


SYNOPSIS
	cgk.sh [-gox] [-sNUM] [AMOUNT] FROM_CURRENCY [VS_CURRENCY]
	cgk.sh -d CRYPTO
	cgk.sh -ee [-pNUM]
	cgk.sh -t CRYPTO [VS_CURRENCY]
	cgk.sh -tt [-pNUM] [CRYPTO]
	cgk.sh -mm [VS_CURRENCY]
	cgk.sh -HH CRYPTO [VS_CURRENCY]
	cgk.sh [-hlv]


DESCRIPTION
	This programme fetches updated crypto and bank currency rates
	from CoinGecko.com and can convert any amount of one supported
	currency into another.

	FROM_CURRENCY is a (crypto)currency or metal symbol. For crypto-
	currencies, coingecko IDs can be used, too. VS_CURRENCY is a
	crypto, fiat or metal symbol and defaults to ${DEFVSCUR,,}.

	List supported currency symbols and IDs with option -l. About
	53 bank currencies (fiat) are supporterd, as well as gold and
	silver. If FROM or VS_CURRENCY is set to any of .=- , it will
	mean the same as bitcoin (and sets scale to 8 in some cases).
	
	Central bank currency conversions are supported indirectly by
	locally calculating them with the cost of one extra call to the
	API. As CoinGecko updates rates frequently (about once a minute),
	it is one of the best APIs for bank currency rates, too.

	Use option -g to convert precious metal rates using grams instead
	of troy ounces (${TOZ} grams / troy ounce). Use option -x to
	use satoshi unit instead of a bitcoin unit (with BTC pairs only).

	Default precision is ${SCLDEFAULTS} and can be adjusted with option -s .
	A shortcut is -NUM in which NUM is an integer. Option -o adds a
	thousands separator (a comma) in result.

	Option -t fetches general of a CRYPTO currency ticker, user may
	set a VS CURRENCY (check available VS_CURRENCIES for this function
	with -mm).

	Option -tt retrieves tickers with CRYPTO from all supported ex-
	changes (note that vs_currency is usd only). *Tip: pipe output
	to grep and fetch specific market pairs.

	Option -b is originally called the bank currency function. It is
	now invoked automatically when needed, however setting it explic-
	itly when appropriate improves script speed. It was first imple-
	mented because of coingecko api restrictions. For example, it
	does not offer rates for conversion between bank currencies and
	metals, neither for some cryptocurrency pairs not officially
	supported by any exchange, so we need do one extra call and
	calculate rates locally.

	Coingecko.com api rate limit is currently 100 requests/minute.
	

ABBREVIATIONS
	Results of some functions may have abbreviations.


		ATH 		All time high
		ATHDelta 	Difference between ATH and current price
		ATL 		All time low
		ATLDelta 	Difference between ATL and current price
		CapTotal 	Total market cap
		CapCh24H 	Market cap change_in the last 24h
		Ch, chg 	Change
		Delta 		Difference
		EX 	 	Exchange name
		EXID 		Exchange identifier
		FDilutdV 	Fully diluted valuation
		INC? 		Incentives for trading?
		MCapRank 	Market capital rank
		Mkt, mk 	Market
		NORM 		Normalised
		R 		Rate
		Rnk 		Rank, trust rank
		ROI 		Return on investiment
		SCORE 		Trust score
		Vol 		Volume


	For more information, such as normal and normalized volume,
	check <https://blog.coingecko.com/trust-score/>.


MARKET CAP FUNCTION 24-H ROLLING STATS -m 
	Some currency convertion data is available for use with the Market
	Cap Function -m. You can choose which currency to display data,
	when available.
	
	Check a list of vs_currencies with option -mm. Otherwise, the
	market capitulation table will display data in various currencies
	in some fields by defaults.


WARRANTY
	Licensed under the GNU Public License v3 or better. It is dis-
	tributed without support or bug corrections. This programme needs
	Bash, cURL or Wget, JQ, Coreutils and Gzip to work properly.

	CoinGecko updates rates at about once per minute.
	
	It is _not_ advisable to depend solely on CoinGecko rates for
	serious trading.
	
	If you found this useful, consider sending me a nickle!  =)

		bc1qlxm5dfjl58whg6tvtszg5pfna9mn2cr2nulnjr


USAGE EXAMPLES		
	(1)     One Bitcoin in US Dollar:
		
		$ cgk.sh btc
		
		$ cgk.sh 1 btc usd


	(2)     100 ZCash in Digibyte with 8 decimal plates:
		
		$ cgk.sh -b -s8 100 zcash digibyte 

		$ cgk.sh -b -8 100 zec dgb

	
	(3)     One thousand Brazilian Real in US Dollar with
		3 decimal plates and using math expression in
		AMOUNT:
		
		$ cgk.sh -b -4 '101+(2*24.5)+850' brl usd 


	(4)    One gram of gold in US Dollars:
				
		$ cgk.sh -g xau usd 


	(5)    Simple ticker for Ethereum against Japanese Yen:

		$ cgk.sh -t eth jpy


	(6)    Tickers of any Ethereum pair from all exchanges:
				
		$ cgk.sh -tt eth 
		
		Tip: pipe to less with opion -S (--chop-long-lines)
		or 'most' pager for scrolling horizontally:

		$ cgk.sh -tt eth | less -S


	(7) 	Market cap function, show data for Chinese CNY:

		$ cgk.sh -m cny


	(8) 	Ten thousand satoshis to N. America USD;
		use satoshi unit, no decimal plates:

		$ cgk.sh -x 10000 btc usd

		same as:
		$ cgk.sh -0 .00010000 btc usd


OPTIONS
	Formatting
	-NUM 	  Same as -sNUM.
	-g 	  Use grams instead of troy ounces (precious metals only).
	-o 	  Print thousands separator in results (comma).
	-s NUM    Scale setting (decimal plates); defaults=${SCLDEFAULTS}.
	-x 	  Use satoshis instead of bitcoins (sat = 1/100,000,000 btc).
	Miscellaneous
	-b 	  Bank currency function, force convertions between
		  unofficially supported currency pairs; defaults=auto.
	-h 	  Show this help.
	-j 	  Debug; print raw data, usually json.
	-v 	  Show this programme version.
	Functions
	-d CRYPTO
		  Dominance of cryptos; a single crypto is optional
		  (amongst top 10 only); also check option -m.
	-e 	  Exchange information; set number of pages with -p.
	-ee	  Print a list of exchange names and IDs only.

	-H CRYPTO [VS_CURRENCY]
		  Historical prices (time series); twice to print csv.
	-l 	  List supported currencies.
	-m [VS_CURRENCY]
		  Market ticker; a vs_currency may be supplied; check
		  option -mm; defaults=USD+others.
	-mm 	  List supported VS_CURRENCIES for options -m and -t .
	-p NUM	  Pages to retrieve (max 100 results/page); use with
		  options -ett; defaults=auto.
	-t CRYPTO [VS_CURRENCY]
		  Simple ticker; general informatioin of CRYPTO.
	-tt [CRYPTO]
		  All tickers with CRYPTO from all exchanges."


#fiat codes
#for quick testing
#removed: btc eth ltc bch bnb eos xrp xlm
FIATCODES=( usd aed ars aud bdt bhd bmd brl cad chf clp cny
	czk dkk eur gbp hkd huf idr ils inr jpy krw kwd
	lkr mmk mxn myr nok nzd php pkr pln rub sar sek
	sgd thb try twd uah vef vnd zar xdr xag xau )  #47


## Functions
## -m Market Cap function		
## -d dominance opt
mcapf()
{
	#discard AMOUNT in $1
	(( $1 )) 2>/dev/null && shift 

	#we don't use $2
	if [[ -n "$2" ]] && [[ "${2,,}" != usd ]]
	then
		echo "Warning: discarding -- $2" >&2
	fi

	# Check if input has a defined vs_currency
	if [[ -z "$1" ]]
	then
		NOARG=1
		set -- "${DEFVSCUR,,}"
	fi

	# Get Data 
	CGKGLOBAL="$TMPD/cgkglobal.json"
	"${YOURAPP3[@]}" "https://api.coingecko.com/api/v3/global" -H  "accept: application/json" >"$CGKGLOBAL"
	VS_CUR=( $( jq -r '.data.total_market_cap|keys[]' "$CGKGLOBAL" ) )

	# Print JSON?
	if (( PJSON )) && (( DOMOPT || MCAP == 2 ))
	then
		cat "${CGKGLOBAL}"
		exit
	fi

	#option -mm
	if (( MCAP == 2 ))
	then
		echo 'Vs_currencies for option -m'
		echo "${VS_CUR[@]^^}"
		exit
	fi

	# Check if input is a valid vs_currency for this function
	if [[ \ "${VS_CUR[*],,}"\  != *\ "${1,,}"\ * ]]
	then
		printf "Not available -- %s\n" "${1^^}" 1>&2
		NOARG=1
		set -- usd
	fi

	#-d only dominance?
	if (( DOMOPT ))
	then
		if [[ -n "$1" ]] && DOM="$( jq -e ".data.market_cap_percentage.${1,,}//empty" "$CGKGLOBAL" )"
		then
			printf "%.${SCL}f %%\n" "${DOM}"
		else
			printf 'Symbol\tDominance%%\n'

			jq -r '.data.market_cap_percentage|to_entries[] | [.key, .value] | @tsv' "$CGKGLOBAL"
			
			totale=( $( jq -r '.data.market_cap_percentage[]' "$CGKGLOBAL" ) )
			printf 'Sum:\t%.4f\n' "$( bc <<< "scale=16; ( ${totale[@]/%/+} 0 ) / 1" )"
			jq -r '(100-(.data.market_cap_percentage|add))' "$CGKGLOBAL" | awk '{ printf "Oth:   %8.4f\n", $1 }'
		fi

		exit
	fi

 	#get more data
 	MARKETGLOBAL="$TMPD/mktglobal.json"
 	DEFIGLOBAL="$TMPD/defiglobal.json"
	"${YOURAPP3[@]}" "https://api.coingecko.com/api/v3/coins/markets?vs_currency=${1,,}&order=market_cap_desc&per_page=10&page=1&sparkline=false" >"$MARKETGLOBAL"
	"${YOURAPP3[@]}" "https://api.coingecko.com/api/v3/global/decentralized_finance_defi" >"$DEFIGLOBAL"

	# Print JSON?
	if (( PJSON ))
	then
		cat "${CGKGLOBAL}"
		
 		echo 'Second json:' >&2
 		cat "${MARKETGLOBAL}"

 		echo 'Third json:' >&2
 		cat "${DEFIGLOBAL}"
		exit
	fi

	#timestamp
	CGKTIME="$( jq -r '.data.updated_at' "$CGKGLOBAL" )"
	
	# Avoid erros being printed
	{
	cat <<-!
	## CRYPTO MARKET STATS
	$( date -d@"$CGKTIME" "+## %FT%T%Z" )
	## Markets_: $( jq -r '.data.markets' "$CGKGLOBAL" )
	## Cryptos_: $( jq -r '.data.active_cryptocurrencies' "$CGKGLOBAL" )
	## ICOs Stats
	 # Upcoming: $( jq -r '.data.upcoming_icos' "$CGKGLOBAL" )
	 # Ongoing_: $( jq -r '.data.ongoing_icos' "$CGKGLOBAL" )
	 # Ended___: $( jq -r '.data.ended_icos' "$CGKGLOBAL" )

	## Dominance (Top 10)
	$( jq -r '.data.market_cap_percentage | keys_unsorted[] as $k | "\($k) \(.[$k])"' "$CGKGLOBAL" | awk '{ printf "  # %s____: %8.4f%%\n", toupper($1), $2 }' )
	$( jq -r '(100-(.data.market_cap_percentage|add))' "$CGKGLOBAL" | awk '{ printf "  # Others_: %8.4f%%\n", $1 }' )
	!

	printf "\n## Total Market Cap\n"
	printf " # Equivalent in\n"
	printf "    %s____: %'22.2f\n" "${1^^}" "$( jq -r ".data.total_market_cap.${1,,}" "$CGKGLOBAL" )"
	if [[ -n "${NOARG}" ]]
	then
		printf "    EUR____: %'22.2f\n" "$( jq -r '.data.total_market_cap.eur' "$CGKGLOBAL" )"
		printf "    GBP____: %'22.2f\n" "$( jq -r '.data.total_market_cap.gbp' "$CGKGLOBAL" )"
		printf "    BRL____: %'22.2f\n" "$( jq -r '.data.total_market_cap.brl' "$CGKGLOBAL" )"
		printf "    JPY____: %'22.2f\n" "$( jq -r '.data.total_market_cap.jpy' "$CGKGLOBAL" )"
		printf "    CNY____: %'22.2f\n" "$( jq -r '.data.total_market_cap.cny' "$CGKGLOBAL" )"
		printf "    XAU(oz): %'22.2f\n" "$( jq -r '.data.total_market_cap.xau' "$CGKGLOBAL" )"
		printf "    BTC____: %'22.2f\n" "$( jq -r '.data.total_market_cap.btc' "$CGKGLOBAL" )"
		printf "    ETH____: %'22.2f\n" "$( jq -r '.data.total_market_cap.eth' "$CGKGLOBAL" )"
		printf "    XRP____: %'22.2f\n" "$( jq -r '.data.total_market_cap.xrp' "$CGKGLOBAL" )"
	fi
	printf " # Change(%%USD/24h): %.4f %%\n" "$( jq -r '.data.market_cap_change_percentage_24h_usd' "$CGKGLOBAL" )"

	cat <<-!

	## Market Cap per Coin
	# SYMBOL      CAP(${1^^})            CHANGE(24h)
	$( jq -r '.[]|"\(.symbol) \(.market_cap)  \(.market_cap_change_percentage_24h)"' "$MARKETGLOBAL"  | awk '{ printf "  # %s  %'"'"'22.2f    %.4f%%\n", toupper($1) , $2 , $3 , $4 }' )
	!

	printf "\n## Market Volume (Last 24H)\n"
	printf " # Equivalent in\n"
	printf "    %s_____ %'22.2f\n" "${1^^}" "$( jq -r ".data.total_volume.${1,,}" "$CGKGLOBAL" )"
	if [[ -n "${NOARG}" ]]
	then
		printf "    EUR____: %'22.2f\n" "$( jq -r '.data.total_volume.eur' "$CGKGLOBAL" )"
		printf "    GBP____: %'22.2f\n" "$( jq -r '.data.total_volume.gbp' "$CGKGLOBAL" )"
		printf "    BRL____: %'22.2f\n" "$( jq -r '.data.total_volume.brl' "$CGKGLOBAL" )"
		printf "    JPY____: %'22.2f\n" "$( jq -r '.data.total_volume.jpy' "$CGKGLOBAL" )"
		printf "    CNY____: %'22.2f\n" "$( jq -r '.data.total_volume.cny' "$CGKGLOBAL" )"
		printf "    XAU(oz): %'22.2f\n" "$( jq -r '.data.total_volume.xau' "$CGKGLOBAL" )"
		printf "    BTC____: %'22.2f\n" "$( jq -r '.data.total_volume.btc' "$CGKGLOBAL" )"
		printf "    ETH____: %'22.2f\n" "$( jq -r '.data.total_volume.eth' "$CGKGLOBAL" )"
		printf "    XRP____: %'22.2f\n" "$( jq -r '.data.total_volume.xrp' "$CGKGLOBAL" )"
	fi
	
	cat <<-!

	## Market Volume per Coin (Last 24H)
	# SYMBOL      VOLUME                  CHANGE
	$( jq -r '.[]|"\(.symbol) \(.total_volume) '${1^^}' \(.market_cap_change_percentage_24h)"' "$MARKETGLOBAL"  | awk '{ printf "  # %s   %'"'"'22.2f %s    %.4f%%\n", toupper($1) , $2 , $3 , $4 }' )

	## Supply and All Time High
	# SYMBOL       CIRCULATING_SUPPLY     TOTAL_SUPPLY
	$( jq -r '.[]|"\(.symbol) \(.circulating_supply) \(.total_supply)"' "$MARKETGLOBAL"  | awk '{ printf "  # %s      %'"'"'22.2f   %'"'"'22.2f\n", toupper($1) , $2 , $3 }' )

	## Price Stats (${1^^})
	$( jq -r '.[]|"\(.symbol) \(.high_24h) \(.low_24h) \(.price_change_24h) \(.price_change_percentage_24h)"' "$MARKETGLOBAL"  | awk '{ printf "  # %s=%s=%s=%s=%.4f%%\n", toupper($1) , $2 , $3 , $4 , $5 }' | column -t -s"=" -N"  # SYMBOL,HIGH(24h),LOW(24h),CHANGE,CHANGE" )

	## All Time Highs (${1^^})
	$( jq -r '.[]|"\(.symbol) \(.ath) \(.ath_change_percentage) \(.ath_date)"' "$MARKETGLOBAL"  | awk '{ printf "  # %s=%s=%.4f%%= %s\n", toupper($1) , $2 , $3 , $4 }' | column -t -s'=' -N'  # SYMBOL,PRICE,CHANGE,DATE' )
	!



	echo -e "\n## Defi Market"
	#defi markets
	jq -r '.data |
		"## MktCap__: \(.defi_market_cap | .[0:18])",
    		"## EthMktCp: \(.eth_market_cap | .[0:18])",
		" # Defi/Eth: \(.defi_to_eth_ratio | .[0:10])",
		" # Dominanc: \(.defi_dominance | .[0:6])",
		" # Vol24h__: \(.trading_volume_24h | .[0:18])",
		" # TopCoin_: \(.top_coin_name)",
		" # TopCDom_: \(.top_coin_defi_dominance | tostring | .[0:6])"' "$DEFIGLOBAL"

	# Avoid erros being printed
	} 2>/dev/null
}

#warning message for more columns
warnf()
{
	echo 'OBS: more columns optionally required' >&2
}

## -e Show Exchange info function
exf() 
{
	# -ee Show Exchange list
	if [[ "${EXOPT}" -eq 2 ]]
	then
		ELIST="$TMPD/elist.json"
		"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/exchanges/list" >"$ELIST"

		# Print JSON?
		if [[ -n ${PJSON} ]]
		then
			cat "${ELIST}"
			exit
		fi
		
		jq -r '.[]|"\(.id)=\(.name)"' "$ELIST" | column -et -s'=' -N"EXCHANGE_ID,EXCHANGE_NAME"
		
		printf 'Exchanges: %s\n' "$(jq -r '.[] | .id? //empty' "$ELIST" | wc -l)"
		
		exit
	fi

	# Test screen width
	# if stdout is redirected; skip this
	if [[ -t 1 ]]
	then
		if [[ "$(tput cols)" -lt 85 ]]
		then
			COLCONF=( -HINC?,COUNTRY,EX -TEXID )
			warnf
		elif [[ "$(tput cols)" -lt 115 ]]
		then
			COLCONF=( -HINC?,EX -TCOUNTRY,EXID )
			warnf
		else
			COLCONF=( -TCOUNTRY,EXID,EX )
		fi
	fi

	#Get pages with exchange info
	# Print JSON?
	if (( PJSON ))
	then
		"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/exchanges?page=1"
		exit
	fi
	
	echo 'Table of Exchanges'

	head="$( "${YOURAPP2[@]}" "https://api.coingecko.com/api/v3/exchanges" 2>&1 | grep -Fie "total:" -e "per-page:" | sort -r )"

	total="$( grep -Fi total <<< "$head"| grep -o '[0-9]*' )"
	ppage="$( grep -Fi page <<< "$head"| grep -o '[0-9]*' )"
	(( TPAGES )) || {
		TPAGES="$(( (total / ppage) + 1 ))"
		(( total % ppage )) || (( TPAGES-- ))
	}

	echo "$head"
	echo 'Columns: Rank,Score,ExchangeID,Vol24hBTC,NormalisedVol,Incentives?,Year,Country,ExchangeName' >&2

	for ((i=TPAGES;i>0;i--))
	do
		#feedback to stderr
		printf '>>p%s/%s\r' "${i}" "${TPAGES}" >&2

		#get data and make table
		"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/exchanges?page=${i}" |
			jq -r 'reverse[] | "\(if .trust_score_rank == null then "??" else .trust_score_rank end)=\(if .trust_score == null then "??" else .trust_score end)=\(.id)=[\(.trade_volume_24h_btc)]=\(.trade_volume_24h_btc_normalized)=\(if .has_trading_incentive == true then "yes" else "no" end)=\(if .year_established == null then "??" else .year_established end)=\(if .country != null then .country else "??" end)=\(.name)"' |
			column -et -s'=' -N"RNK,SCORE,EXID,VOL24H_BTC,NORM_VOL,INC?,YEAR,COUNTRY,EX" ${COLCONF[@]}
	done
	# Check if CoinEgg still has a weird "en_US" in its name that havocks table
}

## Bank currency rate function
bankf()
{
	unset BANK FMET TMET 
	export BANKSKIP=1

	#download currency lists
	clistf
	tolistf
	
	#get data to check currency names and symbols
	local keys=( $( jq -r '.[],keys[]' "$CGKTEMPLIST1" ) )
	#check and change currency ids
	[[ \ "${keys[*]}"\  = *\ "${2,,}"\ * ]] &&
		MAYBE1="$( changevscf "$2" )" &&
		set -- "$1" "$MAYBE1" "$3"
	[[ \ "${keys[*]}"\  = *\ "${3,,}"\ * ]] &&
		MAYBE2="$( changevscf "$3" )" &&
		set -- "$1" "$2" "$MAYBE2"

	if [[ "${2,,}" = xa[ug] ]]
	then
		URI="price?ids=bitcoin,${3,,},${MAYBE2}&vs_currencies=btc,${2,,},${3,,},${MAYBE2}"
	else
		URI="price?ids=bitcoin,${2,,},${3,,},${MAYBE1},${MAYBE2}&vs_currencies=btc,${2,,},${3,,},${MAYBE1},${MAYBE2}"
	fi
	
	# Get CoinGecko JSON
	CGKRATERAW="$TMPD/cgkrateraw.json"
	"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/simple/$URI" >"$CGKRATERAW"

	# Print JSON?
	if (( PJSON ))
	then
		cat "${CGKRATERAW}"
		exit
	fi
	
	# Get rates to from_currency anyways
	if [[ "${2,,}" = btc ]]
	then
		BTCBANK=1
	elif ! BTCBANK="$( mainf 1 "${2,,}" btc 2>/dev/null )"
	then
		BTCBANK="( 1 / $( mainf 1 bitcoin "${2,,}" ) )" || return 1
	fi
	
	# Get rates to vs_currency anyways
	if [[ "${3,,}" = btc ]]
	then
		BTCTOCUR=1
	elif ! BTCTOCUR="$( mainf 1 "${3,,}" btc 2>/dev/null )"
	then
		BTCTOCUR="( 1 / $( mainf 1 bitcoin "${3,,}" ) )" || return 1
	fi

	# Timestamp? No timestamp for this API
	
	# Calculate result
	# Precious metals in grams?
	ozgramf "$2" "$3"
	#satoshi units?
	satoshif "$@" && set -- "${args[@]}"
	
	RESULT="$( bc <<< "scale=16; ( ( ($1) * ${BTCBANK}) / ${BTCTOCUR}) ${GRAM} ${TOZ}" )"
	
	printf "%${OOPT}.*f\n" "${SCL}" "${RESULT}"
}

## -H historical prices/time series -- backup func
## NOT IN USE
#help:-H [PERIOD] [SYMBOL] [VS_CURRENCY]
#	  Historical prices (time series); period may be 24h
#	  (1d), 7d, 14d, 30d, 90d,180d, 1y and max; pass -H
#	  twice to print historical volumes instead of prices.
#histf()
#{
#	#period
#	case "${1:-x}" in
#		7*d*) 	period=7_days ;;
#		14*d*) 	period=14_days ;;
#		30*d*) 	period=30_days ;;
#		90*d*) 	period=90_days ;;
#		180*d*) period=180_days ;;
#		*y*|365*) period=365_days ;;
#		24*h*|1*d*) period=24_hours ;;
#		*) 	period=max ;;
#	esac
#	shift
#
#	#volume or price?
#	if (( OPTHIST == 1 ))
#	then
#		#prices
#		#jq -r '.stats[] | "\(.[1])\t\( (.[0]/1000)|strflocaltime("%Y-%m-%dT%H:%M:%S%Z") )"' <<<"$DATA"
#
#		to="$( date +%s )"
#		if [[ "$period" = max ]]
#		then
#			from=1367107200  #2013-04-28
#		else
#			from="$( date -d "${period/_/ } ago" +%s )"
#		fi
#
#		DATA="$( "${YOURAPP3[@]}" "https://api.coingecko.com/api/v3/coins/${1}/market_chart/range?vs_currency=${2,,}&from=${from}&to=${to}" )"
#		# Print JSON?
#		if (( PJSON ))
#		then
#			echo "$DATA"
#			exit
#		fi
#		[[ -n "$DATA" ]] || exit 1
#
#		jq -r '.prices[] | "\(.[1])\t\( (.[0]/1000)|strflocaltime("%Y-%m-%dT%H:%M:%S%Z") )"' <<<"$DATA"
#	else
#		#volumes
#		ID=( $( "${YOURAPP3[@]}" "https://www.coingecko.com/en/coins/${1}" | sed -nE '/\/[0-9]+\/.*json/  s/.*\/([0-9]+)\/.*/\1/p' ) )
#		#get data
#		DATA="$( "${YOURAPP3[@]}" "https://www.coingecko.com/price_charts/${ID[0]}/${2,,}/${period}.json" )"
#		# Print JSON?
#		if (( PJSON ))
#		then
#			echo "$DATA"
#			exit
#		fi
#		[[ -n "$DATA" ]] || exit 1
#
#		jq -r '.total_volumes[] | "\(.[1])\t\( (.[0]/1000)|strflocaltime("%Y-%m-%dT%H:%M:%S%Z") )"' <<<"$DATA"
#	fi
#}

## -H historical prices/time series
histf()
{
	#get data
	data="$TMPD/data.json"
	"${YOURAPP3[@]}" "https://www.coingecko.com/price_charts/export/${2,,}/${3,,}.csv" >"$data"
	
	#print raw CSV data?
	if (( PJSON )) || (( OPTHIST > 1 ))
	then
		cat "$data"
		exit
	fi

	#make a table
	#header (columns)
	COLS="$( read <"$data" ;echo "$REPLY" )"

	tail -n+2 "$data" | column -ets, -N"${COLS^^}" -TSNAPPED_AT
}

#trap function
trapf()
{
	#unset trap
	trap \  EXIT INT TERM

	[[ -e "$ERRSIG" ]] && RETSIG=1

	[[ -d "$TMPD" ]] && rm -rf "$TMPD"

	exit "${RETSIG:-0}"
}

#-t ticker simple backup func
tickersimplefb()
{
	if [[ "${3,,}" != usd ]]
	then
		echo "Warning: rates against USD only" >&2
	fi
	
	#maybe#https://www.coingecko.com/en/overall_stats
	#downlaod ticker data
	data="$( "${YOURAPP3[@]}" "https://www.coingecko.com/en/coins/${2,,}" )"

	# Print JSON?
	if (( PJSON ))
	then
		echo "$data"
		exit
	fi
	
	#process
	sed -n '/^<p class="mt-5">/,/^\s*$/p' <<<"$data" |
		grep -vie 'http.*://' -e 'Additional information' |
		#user has w3m?
		if command -v w3m &>/dev/null
		then
			w3m -dump -T text/html
		elif command -v elinks &>/dev/null
		then
			elinks -dump
		elif command -v lynx &>/dev/null
		then
			lynx -force_html -stdin -dump
		else
			sed 's/<[^>]*>//g' |
			sed -r '/^ *\S/!b; N; /\n *$/!b; N; /\S *$/!b; s/\n *\n/\n/' |
			awk '!NF {if (++n <= 1) print; next}; {n=0;print}'
		fi
}
#-t default simple ticker function
tickersimplef()
{
	local perpage total i marketglobal data ok
	perpage=600
	total=10

	#get data
	for i in {1..${total}}
	do
		if ! marketglobal="$( "${YOURAPP3[@]}" "https://api.coingecko.com/api/v3/coins/markets?vs_currency=${3,,}&order=market_cap_desc&per_page=${perpage}&page=${i}&sparkline=false" )"
		then
			break
		fi

		# Print JSON?
		if (( PJSON ))
		then
			echo "$marketglobal"
		fi

		if data="$( jq -er ".[] |
			select(.id == \"${2,,}\") //
			select(.symbol == \"${2,,}\") //
			select(.name | ascii_downcase == \"${2,,}\") //
			empty" <<< "$marketglobal" 2>/dev/null )" &&
			[[ -n "$data" ]]
		then
			ok=1
			break
		fi
	done

	#return on error
	(( ok )) || return 1

	#print ticker
	<<<"$data" jq -er '. |
		"Updated_: \(.last_updated)",
		"MCapRank: \(.market_cap_rank)",
		"ID______: \(.id)",
		"Symbol__: \(.symbol)",
		"Name____: \(.name)",
		"CapTotal: \(.market_cap)",
		"CapCh24H: \(.market_cap_change_24h // empty)  \(.market_cap_change_percentage_24h)%",
		"Supply__: \(.circulating_supply // empty)",
		"SupTotal: \(.total_supply // empty)",
		"Sup__Max: \(.max_supply // empty)",
		"ATH_____: \(.ath // empty)",
		"ATHDelta: \(.ath_change_percentage // empty)%",
		"ATH_Date: \(.ath_date // empty)",
		"ATL_____: \(.atl // empty)",
		"ATLDelta: \(.atl_change_percentage // empty)",
		"ATL_Date: \(.atl_date // empty)",
		"ROI_____: times: \(.roi|.times // empty)  cur: \(.roi|.currency)  perc: \(.roi|.percentage)",
		"FDilutdV: \(.fully_diluted_valuation // empty)",
		"PriceNow: \(.current_price)",
		"PriCh24h: \(.price_change_24h // empty)  \(.price_change_percentage_24h)%",
		"High_24H: \(.high_24h // empty)",
		"Low__24H: \(.low_24h // empty)",
		"VolTotal: \(.total_volume // empty)"'  2>/dev/null
}

## -tt Ticker Function from all exchanges
tickerf()
{
	#we don't use $3
	if [[ "${3,,}" != usd ]]
	then
		echo "Warning: discarding -- $3" >&2
	fi

	## Trap temp cleaning functions
	trap trapf EXIT INT TERM
	
	# Test screen width
	# if stdout is redirected; skip this
	if [[ -t 1 ]]
	then
		if [[ "$( tput cols )" -lt 110 ]]
		then
			COLCONF=( -HEX,L_TRADE -TVOL,MARKET,SPD% )
			warnf
		else
			COLCONF=( -TL_TRADE,EX,MARKET,VOL,SPD%,P_USD )
		fi
	fi

	# Start print Heading
	printf "Tickers for %s\n" "${2^^}" 

	#calc how many result pages
	head="$( "${YOURAPP2[@]}" "https://api.coingecko.com/api/v3/coins/${2,,}/tickers" 2>&1 | grep -Fie "total:" -e "per-page:" | sort -r )"

	total="$( grep -Fi total <<<"$head"| grep -o '[0-9]*' )"
	ppage="$( grep -Fi page <<<"$head"| grep -o '[0-9]*' )"
	(( TPAGES )) || {
		TPAGES="$(( (total / ppage) + 1 ))"
		(( total % ppage )) || (( TPAGES-- ))
	}

	echo "$head"
	printf 'Columns: Market,Price,Spread%%,PriceBTC,PriceUSD,Vol,ExchangeID,ExchangeName,LastTrade\n\n' >&2

	# Print JSON?
	if (( PJSON ))
	then
		"${YOURAPP2[@]}" "https://api.coingecko.com/api/v3/coins/${2,,}/tickers"
		"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/coins/${2,,}/tickers?page=${i}"
		exit
	fi

	for (( i=TPAGES ;i>0 ;i--))
	do
		#feedback to stderr
		printf '>>p%s/%s\r' "${i}" "${TPAGES}" 1>&2

		#get data and make table
		"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/coins/${2,,}/tickers?page=${i}" |
			jq -r '.tickers[]|"\(.base)/\(.target)=\(.last)=\(if .bid_ask_spread_percentage ==  null then "??" else .bid_ask_spread_percentage end)=\(.converted_last.btc)=\(.converted_last.usd)=\(.volume)=\(.market.identifier)=\(.market.name)=\(.last_traded_at)"' |
			column -s= -et -N"MARKET,PRICE,SPD%,P_BTC,P_USD,VOL,EXID,EX,L_TRADE" ${COLCONF[@]}
	done
}

## -l Print currency lists
listsf() 
{
	FCLISTS="$TMPD/fclists.json"
	"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/coins/list" >"$FCLISTS"
	VSCLISTS="$TMPD/vsclists.json"
	"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/simple/supported_vs_currencies" >"$VSCLISTS"

	# Print JSON?
	if (( PJSON ))
	then
		cat "${FCLISTS}"
		echo
		cat "${VSCLISTS}"
		exit
	fi

	#check if stdout is free
	if [[ -t 1 ]]
	then
		COLCONF=( -TSYMBOL,ID,NAME )
	fi

	printf "List of supported FROM_CURRENCIES (cryptos)\n"
	jq -r '.[]|"\(.symbol)=\(.id)=\(.name)"' "$FCLISTS" | column -s'=' -et -N'SYMBOL,ID,NAME' ${COLCONF[@]}
	
	printf "\nList of (officially) supported VS_CURRENCY\n"
	jq -r '.[]' "$VSCLISTS" | tr "[:lower:]" "[:upper:]" | sort | column -c 60
	
	printf '\nCriptos: %s\n' "$( jq -r '.[].id' "$FCLISTS" | wc -l )"
	printf 'Fiats/m: %s\n' "$( jq -r '.[]' "$VSCLISTS" | wc -l )"
}

# List of from_currencies
# Create temp file only if not yet created
clistf() 
{
	# Check if there is a list or create one
	if [[ ! -e "$CGKTEMPLIST1" ]]
	then
		# Retrieve list from CGK
		CGKTEMPLIST1="$TMPD/.cgklist1.json"
		"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/coins/list" |
		jq -r '[.[] | { key: .symbol, value: .id } ] | from_entries' >"$CGKTEMPLIST1"
	fi
}

# List of vs_currencies
tolistf()
{
	# Check if there is a list or create one
	if [[ ! -e "$CGKTEMPLIST2" ]]
	then
		# Retrieve list from CGK
		CGKTEMPLIST2="$TMPD/cgklist2.json"
		"${YOURAPP[@]}" "https://api.coingecko.com/api/v3/simple/supported_vs_currencies" |
		jq -r '.[]' >"$CGKTEMPLIST2"
	fi
}

# Change currency code to ID in FROM_CURRENCY
changevscf()
{
	local symbol grepid
	symbol="${1,,}"

	#is a know fiat?
	if [[ \ "${FIATCODES[*]}"\  = *\ "$symbol"\ * ]]
	then
		echo "$symbol"
		return 1
	#try to match in currency list
	elif grepid="$( jq -r ".${symbol}? // empty" "$CGKTEMPLIST1" )" &&
		[[ -n "${grepid//null}" ]]
	then
		echo "$grepid"
		return 0
	else
		echo "$symbol"
		return 1
	fi
}

# Change currency code to ID in FROM_CURRENCY
changetocf()
{
	local grepid id
	id="${1,,}"

	#is a know fiat?
	if [[ \ "${FIATCODES[*]}"\  = *\ "$id"\ * ]]
	then
		echo "$id"
		return 1
	#try to match in currency list
	elif grepid="$( jq -r "to_entries| map(select(.value==\"$id\")) | .[] | .key? // empty" "$CGKTEMPLIST1" )" &&
		[[ -n "${grepid//null}" ]]
	then
		echo "$grepid"
		return 0
	else
		echo "$id"
		return 1
	fi
}

# Precious metals in grams?
ozgramf()
{	
	# Precious metals - troy ounce to gram
	#CGK does not support Platinum(xpt) and Palladium(xpd)
	if (( GRAMOPT ))
	then
		[[ "${2^^}" = XA[GU] ]] && TMET=1
		[[ "${1^^}" = XA[GU] ]] && FMET=1

		if (( TMET - FMET ))
		then
			if (( TMET ))
			then
				GRAM='*'
			else
				GRAM='/'
			fi

			return 0
		fi
	fi

	unset TOZ
	unset GRAM
	return 1
}

#satoshi unit instead of bitcoin unit?
satoshif()
{
	#satoshi opt?
	if (( SATOPT ))
	then
		[[ "${3,,}" =~ ^(bitcoin|btc|bitcoin-cash|bch|bitcoin-cash-sv|bsv)$ ]] && SATTO=1
		[[ "${2,,}" =~ ^(bitcoin|btc|bitcoin-cash|bch|bitcoin-cash-sv|bsv)$ ]] && SATFROM=1
		
		if (( SATTO - SATFROM ))
		then
			if (( SATTO ))
			then
				args=( "( $1 ) * 100000000" "$2" "$3" )
			else
				args=( "( $1 ) / 100000000" "$2" "$3" )
			fi

			return 0
		fi
	fi

	unset args
	return 1
}

#check currencies (defaults)
curcheckf()
{
	## Check VS_CURRENCY
	# Make sure "XAG Silver" does not get translated to "XAG Xrpalike Gene"
	#auto try the bank function, if needed
	if [[ -z "$BANKSKIP$TOPT" ]] &&
		[[ \ "${FIATCODES[*]}"\  = *\ "${2,,}"\ * ]] &&
		bankf "${@}"
	then
		exit 0
	elif clistf &&
		! jq -r '.[],keys[]' "$CGKTEMPLIST1" | grep -qi "^${2}$"
	then
		: >"$ERRSIG"
		printf "ERR: currency -- %s\n" "${2^^}" >&2
		exit 1
	fi
	
	## Check TO_CURRENCY
	if [[ \ "${FIATCODES[*]}"\  != *\ "${3,,}"\ * ]] && {
		[[ -n "${TOPT}" ]] || tolistf && ! grep -qi "^${3}$" "$CGKTEMPLIST2"
	}
	then
		# Bank opt needs this anyways
		#auto try the bank function
		if [[ -z "$BANKSKIP$TOPT" ]] &&
			jq -r '.[],keys[]' "$CGKTEMPLIST1" | grep -qi "^${3}$" &&
			bankf "${@}"
		then
			exit 0
		else
			: >"$ERRSIG"
			printf "ERR: currency -- %s\n" "${3^^}" >&2
			exit 1
		fi
	fi

	return 0
}

#default option - cryptocurrency converter
mainf()
{
	local rate

	#if $CGKRATERAW is set, it is from bank function
	if [[ -f "${CGKRATERAW}" ]]
	then
		# Result for bank subfunction -b
		if rate=$( jq -er '."'${2,,}'"."'${3,,}'" // empty' "$CGKRATERAW" | sed 's/e/*10^/g' ) &&
			rate="$( bc <<< "scale=16 ; ( $1 ) * $rate / 1 " )" &&
			[[ -n "${rate//[a-zA-Z]}" ]]
		then
			echo "$rate"
		else
			((BANKSKIP)) || : >"$ERRSIG"
			return 1
		fi
	else
		# Make equation and print result
		rate="$( "${YOURAPP[@]}" "https://api.coingecko.com/api/v3/simple/price?ids=${2,,}&vs_currencies=${3,,}" )"
		
		#print json?
		if (( PJSON ))
		then
			echo "${rate}"
			exit
		fi
	
		rate="$( jq -r '."'${2,,}'"."'${3,,}'"' <<< "${rate}" | sed 's/e/*10^/g' )"
		
		# Precious metals in grams?
		ozgramf "$2" "$3"
		#satoshi units?
		satoshif "$@" && set -- "${args[@]}"
	
		RESULT="$( bc <<< "scale=16; ( ($1) * ${rate} / 1 ) ${GRAM} ${TOZ}" )"
		
		printf "%${OOPT}.*f\n" "${SCL}" "${RESULT}"
	fi

	return 0
}


# Parse options
while getopts 0123456789bdegxhHljmop:s:tv opt
do
	case ${opt} in
		( [0-9] )
			#scale, same as '-sNUM'
			SCL="${SCL}${opt}"
			;;
		( b )
			## Activate the Bank currency function
			BANK=1
			;;
		( d )
			#single currency dominance
			DOMOPT=1
			MCAP=1
			;;
		( e )
			## List supported Exchanges
			(( EXOPT )) && EXOPT=2 || EXOPT=1
			;;
		( g )
			# Gram opt
			GRAMOPT=1
			;;
		( h )
			# Show Help
			echo -e "${HELP_LINES}"
			exit 0
			;;
		( H )
			#historical prices
			(( OPTHIST )) && OPTHIST=2 || OPTHIST=1
			;;
		( l )
			## List available currencies
			LOPT=1
			;;
		( j )
			# Print JSON
			PJSON=1
			;;
		( m ) 
			## Make Market Cap Table
			(( MCAP )) && MCAP=2 || MCAP=1
			;;
		( o )
			# Thousands sep
			OOPT="'"
			;;
		( p )
			# Number of pages to retrieve with the Ticker Function
			if [[ "$OPTARG" != [0-9]* ]]
			then
				echo "Err: option -p requires an integer -- $OPTARG" >&2
				exit 1
			fi

			TPAGES=${OPTARG}
			;;
		( s )
			# Scale, Decimal plates
			SCL=${OPTARG}
			;;
		( t )
			# Tickers
			(( TOPT )) && TOPT=2 || TOPT=1
			;;
		( v )
			# Version of Script
			head "${0}" | grep -e '# v'
			exit 0
			;;
		( x )
			# Satoshi unit
			SATOPT=1
			;;
		( \? )
			#illegal option
			#printf "Invalid option: -%s\n" "${OPTARG}" >&2
			exit 1
			;;
	esac
done
shift $(( OPTIND - 1 ))

#set exports
export CGKTEMPLIST1 CGKTEMPLIST2 CGKRATERAW

#set scale
SCL_ORIG="$SCL"
if [[ "$SCL" != 0 ]] && ! (( SCL ))
then
	SCL="$SCLDEFAULTS"

	#set result scale to 2 if opt -o is set
	[[ -n "$OOPT" ]] && SCL=2

	#set result scale to nought if opt -x is set
	((SATOPT)) && SCL=0
fi

# Test for must have packages
if ! command -v jq &>/dev/null
then
	printf 'err: JQ is required\n' >&2
	exit 1
fi

#curl or wget
#user agent
UAG='user-agent: Mozilla/5.0 Gecko'
if command -v curl &>/dev/null
then
	YOURAPP=( curl -s -L --compressed --header "$UAG" )
	YOURAPP3=( curl -\# -L --compressed --header "$UAG" )
	YOURAPP2=( "${YOURAPP[@]}" --head )
elif command -v wget &>/dev/null
then
	YOURAPP=( wget -q -O- --header "$UAG" )
	YOURAPP3=( wget -q -O- --show-progress --header "$UAG" )
	YOURAPP2=( "${YOURAPP[@]}" -S )
else
	printf 'err: cURL or Wget is required\n' >&2
	exit 1
fi
unset UAG

#make temp dir
TMPD="$( mktemp -d )" || exit 1
export TMPD

#error filename for exit code
ERRSIG="$TMPD/errsignal.txt"

## Trap temp cleaning functions
trap trapf EXIT INT TERM


# Call opt function
if (( MCAP ))
then
	#market ticker
	mcapf "${@}"
	exit
elif (( EXOPT ))
then
	#list exchanges
	exf
	exit
elif (( LOPT ))
then
	#list currencies
	listsf
	exit
fi

# Set equation arguments
# If first argument does not have numbers
if [[ "$1" != *[0-9]* ]]
then
	set -- 1 "${@}"
# if AMOUNT is not a valid expression for Bc
elif [[ -z "${OPTHIST}${TOPT}" ]] &&
	[[ ! "$1" =~ ^[0-9]+$ ]] &&
	[[ -z "$( bc <<< "$1" )" ]]
then
	printf 'Invalid expression in AMOUNT -- %s\n' "$1" >&2
	exit 1
fi

#set default currencies if none set
[[ -z "$2" ]] &&  	set -- "$1" "${DEFCUR,,}"
[[ -z "$3" ]] && 	set -- "$1" "$2" "${DEFVSCUR,,}"

#change .=- to ``bitcoin''
[[ "$2" = [.=-] ]] && 	set -- "$1" bitcoin "$3"
if [[ "$3" = [.=-] ]]
then
	set -- "$1" "$2" btc
	
	#opt -x has preference over [.=-]
	((SATOPT)) || SCL="${SCL_ORIG:-8}"
fi

#bank opt?
#speed up script a little if these testes are met:
if (( BANK )) ||
	[[ \ "${FIATCODES[*]}"\  = *\ "${2,,}"\ * ]]
then
	bankf "${@}" || {
		: >"$ERRSIG"
		echo "ERR -- check currency codes" >&2
		exit 1
	}
	exit
fi

## Check currencies
#get coin list
clistf

# Check if we can get correct from and to_currency IDs/symbols
set -- "$1" "$(changevscf "$2")" "$(changetocf "$3")"

## Check currency codes
curcheckf "$@"

## Call opt functions
if (( OPTHIST ))
then
	#historical prices
	histf "$@"
elif (( TOPT == 2 ))
then
	#complete ticker from all exchanges
	tickerf "${@}"
elif (( TOPT == 1 ))
then
	#simple ticker
	tickersimplef "$@" || tickersimplefb "$@"
else
	#default option
	#currency conversion
	mainf "$@"
fi

