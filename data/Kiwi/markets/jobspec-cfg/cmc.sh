#!/bin/bash
# cmc.sh -- coinmarketcap.com api access
# v0.11.10  jan/2021  by mountaineerbr

#your cmc api key
#CMCAPIKEY=
export CMCAPIKEY

#defaults
#default from crypto currency
DEFCUR=BTC

#default vs currency
DEFTOCUR=USD

#scale if no custom scale
SCLDEFAULTS=16

#you should not change these:
export LC_NUMERIC=C

#troy ounce to gram ratio
TOZ=31.1034768

#manual and help
#usage: $ cmc.sh [amount] [from currency] [to currency]
HELP_LINES="NAME
	Cmc.sh -- Currency Converter and Market Information
		  Coinmarketcap.com API Access


SYNOPSIS
	cmc.sh [-b] [-gpx] [-sNUM] [AMOUNT] FROM_CURRENCY [TO_CURRENCY]
	cmc.sh -t FROM_CURRENCY [TO_CURRENCY]
	cmc.sh -m [TO_CURRENCY]
	cmc.sh [-adhlv]


DESCRIPTION
	Fetch updated currency rates from <coinmarketcap.com>. This
	programme can convert any amount of one supported crypto cur-
	rency into another. CMC also converts crypto to ~93 central
	bank currencies, gold and silver.

	You can see a list of supported currencies running the script
	with the option -l .

	Central bank currency conversions are supported indirectly by
	locally calculating them with the cost of one extra call. As
	CoinMarketCap updates rates frequently, it is one of the best
	APIs for bank currency rates, too.

	Use option -g to convert precious metal rates using grams
	instead of troy ounces for precious metals, which rate is
	${TOZ}grams/troyounce .  Use option -x to use satoshi unit
	instead of a bitcoin unit (with BTC pairs only).

	Default precision is ${SCLDEFAULTS} and can be adjusted with -s.
	A shortcut is -NUM in which NUM is an integer.

	If you cannot get the rate of a market automatically, try set-
	ting option -b to convert between bank currencies (for ex. usd
	x cad), precious metals rates and other unoficially supported
	(crytpo)currency markets.

	Option -b is originally called the bank currency function. It
	was first implemented because of coinmarketcap api restrictions.
	For example, it does not offer rates for conversion between bank
	currencies and metals, so we need do one extra call and calculate
	rates locally. The resulting rate has great accuracy.

	Rates should be refreshed once per minute by the server.


ENVIRONMENT AND API KEY
	Please take a little time to register at <coinmarketcap.com/api/>
	for a free API key and. Set and export \$CMCAPIKEY variable in
	or set that in the script head source code.

	A builtin key was added for demo purposes, however that may stop
	working at any time or get rate limited quickly.


WARRANTY
	Licensed under the GNU Public License v3 or better. It is
	distributed without support or bug corrections. This programme
	needs Bash, cURL, JQ and Coreutils to work properly.

	It is _not_ advisable to depend solely on <coinmarketcap.com>
	rates for serious trading. Do your own research!

	If you found this script useful, please consider
	sending me a nickle.  =)

		bc1qlxm5dfjl58whg6tvtszg5pfna9mn2cr2nulnjr


USAGE EXAMPLES
	(1)     One Bitcoin in US Dollar:

		$ cmc.sh btc

		$ cmc.sh 1 btc usd


	(2)     One Dash in ZCash, with 4 decimal plates:

		$ cmc.sh -4 dash zec
		
		$ cmc.sh -s4 dash zec


	(3)     One thousand Brazilian Real in US Dollar
		with three decimal plates:

		$ cmc.sh -3b '101+(2*24.5)+850' brl usd


	(4)    One gram of gold in US Dollars:

		$ cmc.sh -g xau usd


	(5) 	Market Ticker in JPY

		$ cmc.sh -m jpy


	(6) 	Ten thousand satoshis to N. America USD;
		use satoshi unit, no decimal plates:

		$ cmc.sh -x 10000 btc usd

		same as:
		$ cmc.sh -0 .00010000 btc usd


OPTIONS
	Miscellaneous
	-a 	  API key status.
	-g 	  Use grams instead of troy ounces; only for precious
		  metals.
	-h 	  Show this help.
	-j 	  Debugging, print JSON.
	-v 	  Script version.
	Formatting
	-NUM 	  Shortcut for scale setting, same as -sNUM .
	-s [NUM]  Set scale (decimal plates) for some opts; defaults=${SCLDEFAULTS}.
	-p 	  Print timestamp, if available.
	Functions
	-b 	  Bank currency function, convertions between
		  unofficially supported currency pairs.
	-d 	  Print dominance stats.
	-l 	  List supported currencies.
	-m [CURRENCY]
		  Rolling 24h market ticker, optionally may set a CURRENCY
		  for converting market stats.
	-t [FROM_CURRENCY] [TO_CURRENCY]
		  Time series (historical prices).
	-x 	  Use satoshis instead of bitcoins (sat = 1/100,000,000 btc)."


METALS='3575=XAU=Gold Troy Ounce
3574=XAG=Silver Troy Ounce
3577=XPT=Platinum Ounce
3576=XPD=Palladium Ounce'

#fiat codes
#use with `test`
FIATCODES=( USD AUD BRL CAD CHF CLP CNY CZK DKK EUR GBP HKD
	HUF IDR ILS INR JPY KRW MXN MYR NOK NZD PHP PKR PLN
	RUB SEK SGD THB TRY TWD ZAR AED BGN HRK MUR RON ISK
	NGN COP ARS PEN VND UAH BOB ALL AMD AZN BAM BDT BHD
	BMD BYN CRC CUP DOP DZD EGP GEL GHS GTQ HNL IQD IRR
	JMD JOD KES KGS KHR KWD KZT LBP LKR MAD MDL MKD MMK
	MNT NAD NIO NPR OMR PAB QAR RSD SAR SSP TND TTD UGX
	UYU UZS VES XAU XAG XPD XPT )

#trap cleaning func
cleanf()
{
	trap \  INT HUP EXIT

	#rm temp dir
	[[ -d "$TMPD" ]] && rm -rf "$TMPD"

	exit 0
}

#check for error response
errf() {
	local ck="$1"

	RESP="$(jq -r '.status|.error_code? // empty' <"$ck")"

	if { [[ -n "$RESP" ]] && ((RESP>0)) ;} ||
		grep -qiE -e 'have been (rate limited|black|banned)' -e 'has banned you' -e 'are being rate limited' "$ck"
	then
		if ! jq -er '.status | .error_message? // empty' <"$ck"
		then
			printf 'Err: run script with -j to check server response\n' >&2
		fi

		#print json?
		if (( PJSON ))
		then
			cat "$ck"
			exit 0
		fi

		return 1
	fi

	return 0
}

#-b bank currency rate function
bankf() {
	unset BANK

	#rerun script, get rates and process data
	if [[ "${2^^}" = BTC ]]
	then
		BTCBANK=( null 1 )
	elif ! BTCBANK=( $( TIMEST=1 GRAM= TOZ= mainf 1 BTC "${2^^}" ) )
	then
		echo "err: invalid currency -- ${2^^}" >&2
		exit 1
	fi
	BTCBANKHEAD="${BTCBANK[0]}" # Timestamp
	BTCBANKTAIL="${BTCBANK[1]}" # Rate

	if [[ "${3^^}" = BTC ]]
	then
		BTCTOCUR=( null 1 )
	elif ! BTCTOCUR=( $( TIMEST=1 GRAM= TOZ= mainf 1 BTC "${3^^}" ) )
	then
		echo "err: invalid currency -- ${3^^}" >&2
		exit 1
	fi
	BTCTOCURHEAD="${BTCTOCUR[0]}" # Timestamp
	BTCTOCURTAIL="${BTCTOCUR[1]}" # Rate

	#print timestamp?
	if (( TIMEST ))
	then
		printf '%s  from_currency\n' "${BTCBANKHEAD}"
		printf '%s  to_currency\n' "${BTCTOCURHEAD}"
	fi

	#precious metals in grams?
	ozgramf "${2}" "${3}"
	#satoshi units?
	satoshif "$@" && set -- "${args[@]}"

	#calculate result & print result
	RESULT="$(bc <<< "scale=16; ( ( (${1}) * ${BTCTOCURTAIL})/${BTCBANKTAIL}) ${GRAM}${TOZ}")"

	#check for errors
	if [[ -z "${RESULT}" ]]
	then
		echo 'err: bad result -- sorry' >&2
		exit 1
	else
		printf "%.${SCL}f\n" "${RESULT}"
	fi
}

#market capital function
mcapf() {
	#check for input to_currency
	if [[ "${1^^}" =~ ^(USD|BRL|CAD|CNY|EUR|GBP|JPY|BTC|ETH|XRP|LTC|EOS|USDT)$ ]]
	then
		true
	elif (( DOMOPT )) || [[ -z "${1}" ]]
	then
		set -- USD
	fi

	#get market data
	CMCGLOBAL="$TMPD/cmcglobal.json"
	curl -s --compressed -H "X-CMC_PRO_API_KEY:  ${CMCAPIKEY}" -H 'Accept: application/json' -d "convert=${1^^}" -G 'https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest' -o "$CMCGLOBAL"

	#print json?
	if (( PJSON ))
	then
		cat "${CMCGLOBAL}"
		exit 0
	fi

	#check error response
	errf "$CMCGLOBAL" || exit 1

	#-d dominance opt
	if [[ -n "${DOMOPT}" ]]
	then
		btc="$(jq -r '.data.btc_dominance' <"$CMCGLOBAL")"
		eth="$(jq -r '.data.eth_dominance' <"$CMCGLOBAL")"
		oth="$( bc <<<"scale=8; 100 - ($btc + $eth)/1" )"
		sum="$( bc <<<"scale=8; ($btc + $eth + $oth)/1" )"

		printf "BTC: %8.4f %%\n" "$btc"
		printf "ETH: %8.4f %%\n" "$eth"
		printf "Oth: %8.4f %%\n" "$oth"
		printf "Sum: %8.4f %%\n" "$sum"
		exit 0
	fi

	#timestamp
	LASTUP="$(jq -r '.data.last_updated' <"$CMCGLOBAL")"

	#avoid erros being printed
	{
	printf '## CRYPTO MARKET INFORMATION\n'
	date --date "${LASTUP}"  '+#  %FT%T%Z'
	printf '\n# Exchanges     : %s\n' "$(jq -r '.data.active_exchanges' <"$CMCGLOBAL")"
	printf '# Active cryptos: %s\n' "$(jq -r '.data.active_cryptocurrencies' <"$CMCGLOBAL")"
	printf '# Market pairs  : %s\n' "$(jq -r '.data.active_market_pairs' <"$CMCGLOBAL")"

	printf '\n## All Crypto Market Cap\n'
	printf "   %'.2f %s\n" "$(jq -r ".data.quote.${1^^}.total_market_cap" <"$CMCGLOBAL")" "${1^^}"
	printf ' # Last 24h Volume\n'
	printf "    %'.2f %s\n" "$(jq -r ".data.quote.${1^^}.total_volume_24h" <"$CMCGLOBAL")" "${1^^}"
	printf ' # Last 24h Reported Volume\n'
	printf "    %'.2f %s\n" "$(jq -r ".data.quote.${1^^}.total_volume_24h_reported" <"$CMCGLOBAL")" "${1^^}"

	printf '\n## Bitcoin Market Cap\n'
	printf "   %'.2f %s\n" "$(jq -r "(.data.quote.${1^^}.total_market_cap-.data.quote.${1^^}.altcoin_market_cap)" <"$CMCGLOBAL")" "${1^^}"
	printf ' # Last 24h Volume\n'
	printf "    %'.2f %s\n" "$(jq -r "(.data.quote.${1^^}.total_volume_24h-.data.quote.${1^^}.altcoin_volume_24h)" <"$CMCGLOBAL")" "${1^^}"
	printf ' # Last 24h Reported Volume\n'
	printf "    %'.2f %s\n" "$(jq -r "(.data.quote.${1^^}.total_volume_24h_reported-.data.quote.${1^^}.altcoin_volume_24h_reported)" <"$CMCGLOBAL")" "${1^^}"
	printf '## Circulating Supply\n'
	printf " # BTC: %'.2f bitcoins\n" "$(bc <<< "scale=16; $(curl -s --compressed "https://blockchain.info/q/totalbc")/100000000")"

	printf '\n## AltCoin Market Cap\n'
	printf "   %'.2f %s\n" "$(jq -r ".data.quote.${1^^}.altcoin_market_cap" <"$CMCGLOBAL")" "${1^^}"
	printf ' # Last 24h Volume\n'
	printf "    %'.2f %s\n" "$(jq -r ".data.quote.${1^^}.altcoin_volume_24h" <"$CMCGLOBAL")" "${1^^}"
	printf ' # Last 24h Reported Volume\n'
	printf "    %'.2f %s\n" "$(jq -r ".data.quote.${1^^}.altcoin_volume_24h_reported" <"$CMCGLOBAL")" "${1^^}"

	printf '\n## Dominance\n'
	printf " # BTC: %'.2f %%\n" "$(jq -r '.data.btc_dominance' <"$CMCGLOBAL")"
	printf " # ETH: %'.2f %%\n" "$(jq -r '.data.eth_dominance' <"$CMCGLOBAL")"

	printf '\n## Market Cap per Coin\n'
	printf " # Bitcoin : %'.2f %s\n" "$(jq -r "((.data.btc_dominance/100)*.data.quote.${1^^}.total_market_cap)" <"$CMCGLOBAL")" "${1^^}"
	printf " # Ethereum: %'.2f %s\n" "$(jq -r "((.data.eth_dominance/100)*.data.quote.${1^^}.total_market_cap)" <"$CMCGLOBAL")" "${1^^}"
	#avoid erros being printed
	} 2>/dev/null
}

#-l print currency lists
listsf() {
	#get data
	PAGE="$TMPD/page.json"
	curl -s --compressed -H "X-CMC_PRO_API_KEY: ${CMCAPIKEY}" -H 'Accept: application/json' -G 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/map' -o "$PAGE"

	#print json?
	if (( PJSON ))
	then
		cat "${PAGE}"
		exit 0
	fi

	#check error response
	errf "$PAGE" || exit 1

	#make table
	printf 'CRYPTOCURRENCIES\n'
	LIST="$(jq -r '.data[] | "\(.id)=\(.symbol)=\(.name)"' <"${PAGE}")"
	column -s'=' -et -N 'ID,SYMBOL,NAME' <<<"${LIST}"

	printf '\nBANK CURRENCIES\n'
	LIST2="$(curl -s --compressed -H "X-CMC_PRO_API_KEY: ${CMCAPIKEY}" -H "Accept: application/json" -d "" -G https://pro-api.coinmarketcap.com/v1/fiat/map | jq -r '.data[]|"\(.id)=\(.symbol)=\(.sign)=\(.name)"')"
	column -s'=' -et -N'ID,SYMBOL,SIGN,NAME' <<<"${LIST2}"
	column -s'=' -et -N'ID,SYMBOL,NAME' <<<"${METALS}"

	printf 'Cryptos: %s\n' "$(wc -l <<<"${LIST}")"
	printf 'BankCur: %s\n' "$(wc -l <<<"${LIST2}")"
	printf 'Metals : %s\n' "$(wc -l <<<"${METALS}")"
}

#time series (historical prices)
timeseriesf()
{
	#user agent
	uag='user-agent: Mozilla/5.0 Gecko'
	#interval
	int=1d
	#end date
	end="$( date -ud00:00:00 +%s )"
	#start date
	start=2013-04-28  #this is the maximum for bitcoin

	#check currencies
	(( $1 )) && shift

	#get coin list
	list="$( PJSON=1 listsf )"

	#get coion id
	id="$( <<<"$list" jq --arg n "${1^^}" --arg s "${1^^}" --arg sl "${1,,}" \
		'.data[] | select(.name == $n // .symbol == $s // .slug == $sl )' |
		jq -r '.id' )"

	tocur="$( <<<"$list" jq --arg n "${2^^}" --arg s "${2^^}" --arg sl "${2,,}" \
		'.data[] | select(.name == $n // .symbol == $s // .slug == $sl )' |
		jq -r '.symbol' )"

	#make sure jq did not return null values
	id="${id//null}"
	tocur="${tocur//null}"

	#check id or return
	(( id )) || return 1

	#get data
	data="$TMPD/timeseries.json" 
	curl -sL --compressed --header "$uag" "https://web-api.coinmarketcap.com/v1.1/cryptocurrency/quotes/historical?convert=${tocur:-${2^^}}&format=chart_crypto_details&id=${id}&interval=${int}&time_end=${end}&time_start=${start}" -o "$data"

	#print json?
	if (( PJSON ))
	then
		cat "$data"
		exit 0
	fi

	#extract relevant data
	keys="$( jq -r '.data | keys_unsorted[]' <"$data")"
	prices="$( jq -r '.data[] | .[] |.[0] ' <"$data")" 
	vol24h="$( jq -r '.data[] | .[] |.[1] ' <"$data")" 
	mcap="$( jq -r '.data[] | .[] |.[2] ' <"$data")" 

	#make table
	paste <(echo "$prices") <(echo "$mcap") <(echo "${vol24h}") <(echo "$keys") |
		column -et -NPRICE,MCAP,24HVOL,DATE
}

#-a api status
apif() {
	PAGESTAT="$TMPD/stat.json"
	curl -s --compressed -H "X-CMC_PRO_API_KEY: ${CMCAPIKEY}" -H 'Accept: application/json'  'https://pro-api.coinmarketcap.com/v1/key/info' -o "$PAGESTAT"

	#print json?
	if (( PJSON ))
	then
		cat "$PAGESTAT"
		exit 0
	fi

	#check error response
	errf "$PAGESTAT" || exit 1

	#print heading and status page
	printf 'API key: %s\n\n' "${CMCAPIKEY}"
	
	jq <"$PAGESTAT"
}

#precious metals in grams?
ozgramf() {
	#precious metals - ounce to gram
	if (( GRAMOPT ))
	then
		[[ "${2^^}" = X[AP][UGTD] ]] && TMET=1
		[[ "${1^^}" = X[AP][UGTD] ]] && FMET=1

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

#defaults func -- currency converter
mainf()
{
	CMCJSON="$TMPD/cmc_${RANDOM}.json"
	curl -s --compressed -H "X-CMC_PRO_API_KEY: ${CMCAPIKEY}" -H 'Accept: application/json' -d "&symbol=${2^^}&convert=${3^^}" -G 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest' -o "$CMCJSON"
	
	#print json?
	if (( PJSON ))
	then
		cat "$CMCJSON"
		exit 0
	fi

	#get pair rate
	CMCRATE="$(jq -r ".data[] | .quote.${3^^}.price" <"${CMCJSON}" | sed 's/e/*10^/g')"
	
	#print json timestamp ?
	if (( TIMEST ))
	then
		JSONTIME="$(jq -r ".data.${2^^}.quote.${3^^}.last_updated" <"${CMCJSON}")"
		if [[ -n "$JSONTIME" ]]
		then
			date --date "$JSONTIME" +%FT%T%Z
		else
			echo no_timestamp
		fi
	fi
	
	RESULT="$(bc <<< "scale=16; ( ( ${1} ) * ${CMCRATE} ) ${GRAM}${TOZ}" )"

	[[ -z "$RESULT" ]] && return 1
	
	printf "%.${SCL}f\n" "${RESULT}"
}


#parse options
while getopts :0123456789abdlmghjs:ptvx opt
do
	case ${opt} in
		( [0-9] ) #scale, same as '-sNUM'
			SCL="${SCL}${opt}"
			;;
		( a ) #api key status
			APIOPT=1
			;;
		( b ) #hack central bank currency rates
			BANK=1
			;;
		( d ) #dominance only opt
			DOMOPT=1
			MCAP=1
			;;
		( g ) #gram opt
			GRAMOPT=1
			;;
		( j ) #debug: print json
			PJSON=1
			;;
		( l ) #list available currencies
			LISTS=1
			;;
		( m ) #market capital function
			MCAP=1
			;;
		( h ) #show help
			echo "${HELP_LINES}"
			exit 0
			;;
		( p ) #print timestamp with result
			TIMEST=1
			;;
		( s ) #decimal plates
			SCL="${OPTARG}"
			;;
		( t ) #time series (historical prices)
			OPTHIST=1
			;;
		( v ) #script version
			grep -m1 '# v' "${0}"
			exit 0
			;;
		( x ) #satoshi units
			SATOPT=1
			;;
		( \? )
			printf 'Invalid option: -%s\n' "$OPTARG" 1>&2
			exit 1
			;;
	esac
done
shift $((OPTIND -1))

export SYMBOLLIST FIATLIST CCHECK

#api key test
if [[ -z "${CMCAPIKEY}" ]]
then
	#dev demo key
	CMCAPIKEY=30394e61-6fdf-4552-8f95-7e228b598f98
fi

#test for must have packages
if (( CCHECK ))
then
	if ! command -v jq &>/dev/null
	then
		echo 'JQ is required' >&2
		exit 1
	fi
	if ! command -v curl &>/dev/null
	then
		echo 'cURL is required' >&2
		exit 1
	fi
	CCHECK=1
fi

#set custom scale
#set scale
if [[ "$SCL" != 0 ]] && ! (( SCL ))
then
	SCL="$SCLDEFAULTS"

	#set result scale to nought if opt -x is set by defaults
	((SATOPT)) && SCL=0
fi

#make temo dir, lots of data
trap cleanf INT HUP EXIT
TMPD="$(mktemp -d)" || exit 1

#call opt functions
if (( MCAP ))
then
	mcapf "${@}"
	exit
elif (( LISTS ))
then
	listsf
	exit
elif (( APIOPT ))
then
	apif
	exit
fi

#set equation arguments
#if first argument does not have numbers
if [[ "${1}" != *[0-9]* ]]
then
	set -- 1 "${@}"
#if amount is not a valid expression for bc
elif [[ ! "$1" =~ ^[0-9]+$ ]] &&
	[[ -z "$(bc <<< "$1")" ]]
then
	printf 'Err: invalid expression in AMOUNT\n' 1>&2
	exit 1
fi
if [[ -z ${2} ]]
then
	set -- "${1}" ${DEFCUR^^}
fi
if [[ -z ${3} ]]
then
	set -- "${1}" "${2}" ${DEFTOCUR^^}
fi

#call opt -- historical price
if (( OPTHIST))
then
	timeseriesf "$@"
	exit
fi

#currency converter
if (( BANK )) ||
	[[ \ "${FIATCODES[*]}"\  = *\ ${2^^}\ * ]]
then
	#try the bank function
	bankf "${@}"
	exit
fi

#default markets -- get data
#if error response
if ! errf "${CMCJSON}" 2>/dev/null
then
	#try the bank function
	bankf "${@}"
	exit
else
	#make equation and calculate result
	#metals in grams?
	ozgramf "${2}" "${3}"
	#satoshi units?
	satoshif "$@" && set -- "${args[@]}"
	
	#defaults func
	mainf "$@"
fi

