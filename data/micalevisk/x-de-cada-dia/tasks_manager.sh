#!/bin/bash
##
##  v0.21-2
##  resources for cursor movements with ANSI escape sequences and other stuffs:
##  - http://shellscript.com.br
##  - https://www.tldp.org/HOWTO/Bash-Prompt-HOWTO/x361.html
##  - https://www.youtube.com/watch?v=K_6peGEsq0U
##  - https://github.com/piotrmurach/tty-cursor/tree/master/spec/unit
##  - https://unix.stackexchange.com/questions/88296
##  - https://docs.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences
##  - https://stackoverflow.com/questions/32009787
##  - http://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm
##  - https://developer.apple.com/library/content/documentation/OpenSource/Conceptual/ShellScripting/AdvancedTechniques/AdvancedTechniques.html
##  - https://www.gnu.org/software/bash/manual/html_node/Job-Control.html
##  - http://aurelio.net/sed/sed-howto/#fluxo-texto
##
##  Programa pensado na seguinte ordem de prioridade das operações:
##    marcar como feita - adicionar nova tarefa - remover tarefa - editar tarefa
##
##  known issues:
##  1- O prompt não é mostrado corretamente se a linha do comando que executou o script não for a primeira da janela do terminal;
##  2- Ao fechar o editor de texto (após a ação 'edit task') o prompt pode voltar com algumas linhas apagadas;
##  3- Não apaga o diretório/arquivo criado (antes) para uma tarefa caso o usuário deseja (agora) removê-la;
##  4- Não considera (corretamente) o redimensionamento da janela;
##  5- Não lista corretamente se o número de linhas da janela for menor que o número de itens;
##  6- Não associa arquivos/diretórios criados fora da mesma execução em que a tarefa foi marcada como feita;
##

shopt -s extglob ## ativar extended pattern matching features ~ remover esta linha se for zsh

declare READ_WITH_PROMPT="read -p"
declare EDITOR="vim"
declare OPEN="firefox"
declare EXPLORER="explorer" ## `xdg-open` or `nauttilus` to open a file manager
command -v $EDITOR >/dev/null 2>&1 || EDITOR="vi" ## or `open -e`
command -v $OPEN >/dev/null 2>&1 || OPEN="cygstart"
command -v $EXPLORER >/dev/null 2>&1 || EXPLORER="xdg-open"

# exec 2>/dev/null ## não mostrar erros na saída padrão; causa ERRO no tput no MinGW

# ------------------------------------------------------------------------------------------------------------------------------- #
declare -A COLORS=( [w]=$'\e[37;1m' [y]=$'\e[33m' [g]=$'\e[32m' [r]=$'\e[31m' [p]=$'\e[35;1m' [n]=$'\e[0m' [gr]=$'\e[30;1m' )
declare -A TASK_REF_EMOJIS=( [d]="file_folder" [f]="memo")
declare -a HEADERS_E=("<!-- title* -->\n" "<!-- last update -->\n" "<!-- snippet -->\n" "<!-- notes -->\n")
declare -r NAVI_SYMBOL='\xE2\x96\xB8'
declare -r NAVI_LENGTH=2 ## forçar tamanho aqui se for usar símbolo hexadecimal (UTF-8 literal)
declare -r NAVI_COLOR=${COLORS[y]/%m/;1m}
declare -r NAVI_COLUMN=0
declare -r SEPARATOR='.'
declare -r TASKS_FILE="README.md"
declare -r MISCELLANEOUS_DIRNAME="avulsos"
declare -r TASK_DONE_MARK=":white_check_mark:"
declare -r SPECIAL_LIST="[]\[|\`_*]" ## lista regex dos caracteres que foram escapados no title para evitar a interpretação do MD
declare -r KERNEL_NAME="$(uname -s)"
declare -r NOT_MINGW_TERM="${KERNEL_NAME/#MINGW[[:digit:]]*}"
declare -r CURR_DIR="${1%%/}"
declare -r PATH_TO_TASKS_FILE="${CURR_DIR,,}/$TASKS_FILE"
declare -r PATH_TO_MISCELLANEOUS_DIRNAME="${CURR_DIR,,}/$MISCELLANEOUS_DIRNAME"
declare -A tasks_done ## a chave é o seu index (no array de tarefas) e o valor é a sua linha real
declare -A tasks_to_remove ## a chave é o seu index (no array de tarefas) e o valor é a sua linha real
declare -A tasks_edit ## a chave é o seu index (no array de tarefas) e o valor é a sua linha real
declare -A created_files ## a chave é a linha real da tarefa e o valor é o caminho para o arquivo
declare -A created_dirs ## a chave é a linha real da tarefa e o valor é o caminho para o diretório
declare -a list_items ## cada elemento é um item na lista que será exibida
declare -i index
declare -i num_items
declare -i offset
declare -i column_sep
declare -i screen_width
declare -i curr_item_index
declare -l normalized_task_name
declare -l file
declare -l dir
declare SOF_CURSOR_POS
declare eof_cursor_pos
declare all_tasks
declare MODE_INIT
declare MODE_EDIT
declare MODE_NEW
declare command_end_action
# ------------------------------------------------------------------------------------------------------------------------------- #

trap update_screen_width__ WINCH ## user has resized the window
trap clear_screen_exit SIGINT  ## user press Ctrl-C
trap clear_screen_exit SIGTSTP SIGHUP SIGKILL SIGQUIT SIGTERM ## user press Ctrl-Z or quit this process

# ===================================================== #
# ===================== begin ========================= #
# @use: grep sed gawk read mapfile                      #
# ===================================================== #
main__() {
  local items_to_show
  local bind_arrow_right bind_arrow_left bind_delete bind_d bind_f bind_o

  commands_switcher__ "$@"

  if [ -n "$MODE_INIT" ]; then ## caso especial: não usa a interação avançada
    command_init "${CURR_DIR,,}"
  fi

  clear ## apagar tela para previnir a issue#1 ~ movee o cursor para (1,1)
  SOF_CURSOR_POS="0;0"
  # printf "\\e[6n"; read -sdR SOF_CURSOR_POS

  ## definindo bindings padrões
  bind_delete=toggle_delete_task
  bind_d=create_dir
  bind_f=create_file
  bind_o=open_title_link
  bind_dot=open_dir

  if [ -n "$MODE_EDIT" ]; then ## items as all tasks
    set_all_tasks__
    items_to_show="$(printf "$all_tasks" | sort -n | sed -E "s~\\\(${SPECIAL_LIST})~\1~g")" ## lista de tarefas encontradas

    bind_arrow_right=mark_edit_task
    bind_arrow_left=remove_edit_mark
  elif [ -n "$MODE_NEW" ]; then ## items as all sections
    items_to_show="$(sed -En '/^##\s*(.+)/=; s//\1/p' "$PATH_TO_TASKS_FILE" | sed 'N; s/\n/:/')" ## lista das seções econtradas

    bind_arrow_right=bind_blank
    bind_delete=; bind_d=; bind_f=; bind_o=; ## desativando bindings padrões
  else ## items as tasks not done
    items_to_show="$(grep --color=never -n -o -P '(?<=^\|\| \[).+?[^\\](?=\])' "$PATH_TO_TASKS_FILE" | sed -E "s~\\\(${SPECIAL_LIST})~\1~g")" ## lista de tarefas pendentes

    bind_arrow_right=mark_done_task
    bind_arrow_left=remove_done_mark
  fi


  mapfile -t list_items <<< "$items_to_show" ## array com elementos no formato `<real_line>:<task_tittle>`
  [ -n "${list_items[0]//[[[:cntrl:]][[:blank:]]]}" ] || bind_blank ## verifica se existe algo para listar

  num_items=${#list_items[@]} ## quantidade de linhas obtidas da extração das tarefas
  offset=$(( NAVI_LENGTH + ${#num_items} )) ## quantidade de colunas antes do 'SEPARATOR'
  column_sep=$(( offset + 1 ))

  update_screen_width__
  [ $(tput lines) -lt $num_items ] && resize_window $(( num_items + 5 )) ## previne (parcialmente) a issue#5
  ## HACK:                                                         ^^ depende do número de lihhas exibidas

  save_cursor
  gawk -v offset=$offset -v sep="$SEPARATOR" '{ printf "%*d%s %s\n", offset, NR, sep, gensub(/^[0-9]+:?/, "", 1) }' <<< "$items_to_show"
  update_eof_cursor_pos__
  restore_cursor
  hide_cursor

  curr_item_index=0
  move_to_column $column_sep
  print_navi

  while read -rsn1 ui; do
    case "$ui" in
      $'\x1b') ## handle ESC sequence.
        read -rsn2 -t 0.1 key
        [ "${key:0:1}" != '[' ] && bind_esc || {
          case "${key:1:1}" in
            'A') bind_arrow_up ;;
            'B') bind_arrow_down ;;
            'C') $bind_arrow_right ;;
            'D') $bind_arrow_left ;;
            '3') $bind_delete ;;
          esac
        }
        ## flush "stdin" with 0.1  sec timeout.
        read -rsn5 -t 0.1 ;;

      $'') bind_blank ;;

      ## handle especial alpha chars.
      $'\x64') $bind_d ;;
      $'\x66') $bind_f ;;
      $'\x6f') $bind_o ;;
      $'\x2e') $bind_dot ;;

      *) ;; ## do nothing
    esac
  done
}


# ====================================================== #
# ====================== util ========================== #
# @use: read sed                                         #
# ====================================================== #

## Exibe um mensagem que espera a tecla 'y' (1 caractere) via stdin.
## @args: <pergunta que será feita>
## @use: read
confirm() {
  read -n1 -p "$1? ${COLORS[r]}[y/N]${COLORS[n]} " 2>&1
  [ "${REPLY,,}" == "y" ]
}

## Define as variáveis 'file' e 'dir' que são caminhos para o
## arquivo ou diretório da tarefa. Além da variável
## 'normalized_task_name' que guarda o nome tratado da tarefa.
## Esses caminhos são em relação a raiz do projeto e serão usados para
## criar um arquivo/dir.
## @args: <nome da tarefa> [extensão para o arquivo]
## @use: sed
set_file_and_dir__() {
  ## XXX: talvez não precise remover a acentuação pois o file system as permite.
  normalized_task_name=$(sed -E '
    y/àáâãäåèéêëìíîïòóôõöùúûü/aaaaaaeeeeiiiiooooouuuu/
    y/ÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜ/AAAAAAEEEEIIIIOOOOOUUUU/
    y/çÇñÑß¢Ðð£Øø§µÝý¥¹²³ªº/cCnNBcDdLOoSuYyY123ao/
    y/ /_/
    s_[:?"*<>|\/#]__g
    s_([^[:alnum:]])\1_\1_g' <<< "${1,,}")

  file="$PATH_TO_MISCELLANEOUS_DIRNAME/${normalized_task_name}.$2"
  dir="$CURR_DIR/$normalized_task_name"
}

## Define a variável 'all_tasks' que é uma string com todos os `title`
## recuperados do arquivo das tarefas.
## @use: grep
set_all_tasks__() {
  ## OPTIMIZE: evitar o uso de dois grep (talvez com sed -En e o comando '='; para casar tudo e apagar lixos).
  all_tasks_done="$(grep --color=never -n -o -P "(?<=${TASK_DONE_MARK} \| \[).+?[^\\\](?=\])" "$PATH_TO_TASKS_FILE")"
  all_tasks="${all_tasks_done:+${all_tasks_done}\n}$(grep --color=never -n -o -P '(?<=^\|\| \[).+?[^\\](?=\])' "$PATH_TO_TASKS_FILE")"
}

show_help_exit() {
  cat <<-EOF
  "Fancy" CLI Tool para gerenciar as tarefas de \`X de Cada Dia\`.

  Usage:
    $0 <path/to/lang/dir> [COMMANDS]

  A ausência de um comando implicará na listagem das tarefas pendentes
  que poderão sofrer as seguintes ações:
    .____._________________._____________________________________________________________________.
    | n. | key             |                             description                             |
    +----+-----------------+---------------------------------------------------------------------+
    | 1  | ESC             | cancel actions and exit                                             |
    | 2  | down arrow key  | next task                                                           |
    | 3  | up arrow key    | previous task                                                       |
    | 4  | right arrow key | mark as done                                                        |
    | 5  | left arrow key  | unmark as done                                                      |
    | 6  | delete          | mark/unmark to remove                                               |
    | 7  | d               | create a [d]irectory to current task (emit 'beep' if it was created)|
    | 8  | f               | create a [f] to current task (waiting for an extension after 'beep')|
    | 9  | o               | [o]pen task \`title\` link                                            |
    | 10 | space/ENTER     | go to next step (and update the progress)                           |
    | 11 | .               | open current project directory                                      |
    +----+-----------------+---------------------------------------------------------------------+

  COMMANDS:
    init   - Create a new directory with proper 'README.md'. (non-interactive)
    new    - Select the section to create a new task. (off: 4..9)
    edit   - Select tasks to edit.
EOF
  exit 0
}


# ===================================================== #
# ==================== command ======================== #
# @use: mktemp sed rm mapfile less                      #
# ===================================================== #

## Solicita informações para a criação
## de um novo diretório (para uma nova lang) com
## o boilerplate do arquivo `README.md` adequado.
## @args: <diretório a ser criado>
command_init() {
  DEFAULT_BADGE_WIDTH="180"
  DEFAULT_LOGO_WIDTH="160"

  new_dir="${1,,}" ## Language name
  new_default_file="${new_dir}/README.md"

  [ -e "$new_dir" ] && { echo "[ERROR] File '${new_dir}' exists." && exit 1; }

  $READ_WITH_PROMPT "Badge Hex Color (look https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml): " lang_badge_color && lang_badge_color=${lang_badge_color//#}
  [ -z $lang_badge_color ] && { echo "[ERROR] Badge color is undefined." && exit 2; }

  $READ_WITH_PROMPT "Badge Width (default=${DEFAULT_BADGE_WIDTH}): " lang_badge_width && lang_badge_width=${lang_badge_width:-$DEFAULT_BADGE_WIDTH}
  $READ_WITH_PROMPT "Logo Path: " lang_logo_path && [ -z $lang_logo_path ] && { echo "[ERROR] Logo path is undefined." && exit 3; }
  $READ_WITH_PROMPT "Logo Width (default=${DEFAULT_LOGO_WIDTH}): " lang_logo_width && lang_logo_width=${lang_logo_width:-$DEFAULT_LOGO_WIDTH}

  mkdir -p "$new_dir" || return 1 ## ERROR

  ## FIXME: trailling tabs are not stripped
  cat <<-EOF > "$new_default_file"
  <div align="center">
    <img src="${lang_logo_path}" width="${lang_logo_width}">
    <h1><i>${lang_name}</i> de Cada Dia</h1>
    <img src="https://img.shields.io/badge/done-0%25%20(0%20of%200)-${lang_badge_color}.svg" width="${lang_badge_width}">
  </div>

  <p align="center">
    <a href="#vídeos">:video_camera:</a>&nbsp;
    <a href="#screencasts-e-relacionados">:floppy_disk:</a>&nbsp;
    <a href="#artigos-e-relacionados">:newspaper:</a>&nbsp;
    <a href="#livros">:books:</a>&nbsp;
  </p>

  ---

  <div align="center">


  ## Vídeos

  status | title | last update | snippet | notes
  :-----:|:------|:-----------:|:-------:|:----:


  ## Screencasts e Relacionados

  status | title | last update | snippet | notes
  :-----:|:------|:-----------:|:-------:|:----:


  ## Artigos e Relacionados

  status | title | last update | snippet | notes
  :-----:|:------|:-----------:|:-------:|:----:


  ## Livros

  status | title | last update | snippet | notes
  :-----:|:------|:-----------:|:-------:|:----:


  </div>
EOF

  printf "~ Diretório %s${new_dir}%s Criado com o Arquivo %s${new_default_file}\\n" ${COLORS[gr]} ${COLORS[n]} ${COLORS[g]}

  exit 0
}

## Verifica se existe uma tarefa com o title passado.
## @args: <title da tarefa>
task_exists() {
  ## FIXME: procura apenas a ocorrência do título em qualquer sequência
  [ $(grep -c -m1 -F "$1" "$PATH_TO_TASKS_FILE") -gt 0 ]
}

## Abre um editor de texto com as informações da tarefa
## passa por parâmetro a serem editadas e persiste.
## @args: <linha real da tarefa>
edit_file() {
  local new_values task_title
  local temp_file="$(mktemp -q ".edit-task-L${1}.XXXXXXXXXXXX.md")"
  [ -w "$temp_file" ] || return 1 ## ERROR

  ##<%e
  set_all_tasks__
  less < <(printf "Todas as Tarefas:\\n${all_tasks}" | sort -n | sed -E "s~\\\(${SPECIAL_LIST})~\1~g") ## lista todas as tarefas
  ##e%>

  sed -En "${1}\
    s/^(\|\||${TASK_DONE_MARK}\s*\|)\s*(.+)\|\s*(.*)\|\s*(.*)\|\s*(.*)/${HEADERS_E[0]}\2\n\n${HEADERS_E[1]}\3\n\n${HEADERS_E[2]}\4\n\n${HEADERS_E[3]}\5/p"\
    "$PATH_TO_TASKS_FILE" > "$temp_file" || return 2 ## ERROR

  ##<%d
  $EDITOR -c 'set nu' +2 "$temp_file" ## abrir arquivo para edição e posicionar na segunda linha

  ## Admite que o padrão do arquivo aberto não será alterado e todos os valores
  ## estão 1 linha abaixo do campo indicado e não ocupam mais de 1 linha.
  mapfile -t new_values < <(sed -n '2p; 5p; 8p; 11p' "$temp_file") ## XXX: versão "bruta" mas rápida
  rm -f "$temp_file"
  task_title="${new_values[0]##*( )}" ## trim leading whitespaces
  [ -n "$task_title" ] || return 3 ## ERROR 'title' (obrigatório) está vazio

  ## formatando como colunas
  new_values="${new_values[@]/%/\|}"
  new_values="${new_values%\|}"
  new_values="${new_values%%*( )}" ## trim trailing whitespaces
  new_values="$(sed -E "s~(\\\)(${SPECIAL_LIST})~\1\1\2~g" <<< "$new_values")"

  task_exists "${task_title}" && return 4
  ##d%>

  sed -Ei "${1} s~^(\|\||${TASK_DONE_MARK}\s*\|)(\s*).+~\1\2${new_values}~" "$PATH_TO_TASKS_FILE" && emit_beep
}

## Itera sobre as tarefas marcadas para edição
## executando a ação de editar arquivo.
command_edit() {
  for i in "${!tasks_edit[@]}"; do
    edit_file "${tasks_edit[$i]}" && printf "Tarefa %s$(( i + 1 ))%s editada!\\n" ${COLORS[y]} ${COLORS[n]}
  done
}

## Abre um editor de texto com as informações necessárias
## para a inserção de uma nova tarefa em uma seção específica.
## @args: <linha real de seção>
create_new_task() {
  local new_task_values task_title line_first_task
  local temp_file="$(mktemp -q ".new-task-L${1}.XXXXXXXXXXXX.md")"
  [ -w "$temp_file" ] || return 1 ## ERROR

  ##<%e
  set_all_tasks__
  less < <(printf "Todas as Tarefas:\\n${all_tasks}" | sort -n | sed -E "s~\\\(${SPECIAL_LIST})~\1~g") ## lista todas as tarefas
  ##e%>

  line_first_task="$(sed -En "$(( $1 + 2 )),\
    /^\s*(${TASK_DONE_MARK}|\|\|)/{
      /^\s*(${TASK_DONE_MARK}|\|\|)/=;
    }" "$PATH_TO_TASKS_FILE")"

  printf "%s\\n\\n\\n" "${HEADERS_E[@]//\\n}" > "$temp_file" || return 2 ## ERROR

  ##<%d
  $EDITOR -c 'set nu' +2 "$temp_file" ## abrir arquivo para edição e posicionar na segunda linha

  ## Admite que o padrão do arquivo aberto não será alterado e todos os valores
  ## estão 1 linha abaixo do campo indicado e não ocupam mais de 1 linha.
  mapfile -t new_task_values < <(sed -n '2p; 5p; 8p; 11p' "$temp_file") ## XXX: versão "bruta" mas rápida
  rm -f "$temp_file"

  task_title="${new_task_values[0]##*( )}" ## trim leading whitespaces
  [ -n "$task_title" ] || return 3 ## ERROR 'title' (obrigatório) está vazio

  ## formatando como colunas
  new_task_values="${new_task_values[@]/%/ \|}"
  new_task_values="${new_task_values%\|}"
  new_task_values="${new_task_values%%*( )}" ## trim trailing whitespaces
  new_task_values="$(sed -E "s~(\\\)(${SPECIAL_LIST})~\1\1\2~g" <<< "$new_task_values")"

  task_exists "$task_title" && return 4
  ##d%>

  sed -i "${line_first_task}i\\|| $new_task_values" "$PATH_TO_TASKS_FILE"
}

## Trata a seção corrente como a alvo para a nova tarefa.
## Chama a ação de criar tarefa.
command_new() {
  ## A tarefa será inserida no início, i.e., primeira item da seção escolhida
  if create_new_task "${list_items[$curr_item_index]%%:*}"; then
    printf "Tarefa Inserida em %s${list_items[$curr_item_index]#*:}%s!\\n" ${COLORS[y]} ${COLORS[n]}
  else
    printf "Não foi possível inserir essa tarefa...\\n"
  fi
}


## Controla o uso dos comandos. Aceita um comando por
## execução. Trata apenas o primeiro, caso mais de
## um forem fornecidos.
commands_switcher__() {
  [ $# -lt 2 ] && return 1 ## ERROR ~ nenhum comando encontrado, ignorar
  case "${2,,}" in
    init) MODE_INIT=1 ;;
    edit) MODE_EDIT=1 ;;
    new ) MODE_NEW=1  ;;
    * ) show_help_exit;;
  esac
}


# ===================================================== #
# ==================== binding ======================== #
# @use: sed printf mkdir read touch open                #
# ===================================================== #

bind_arrow_up() { previous_item__; }
bind_arrow_down() { next_item__; }
bind_esc() { clear_screen_exit; }

## Etapa final que lista os diretórios e arquivos criados,
## espera a confirmação das ações realizadas e atualiza o
## arquivo das tarefas (progresso, etc).
bind_blank() {
  local num_tasks num_tasks_done percentage
  local color_key="r"

  move_to_eof || {
    move_to_sof ## para a versão no MinGW
    printf "\\e[0J" ## apagar até o fim da tela
  }

  show_cursor

  [ -n "$num_items" ] || exit 0

  if [ -n "$MODE_EDIT" ]; then
    num_tasks=$num_items
    num_tasks_done=$(grep -c -P "^\s*${TASK_DONE_MARK}" "$PATH_TO_TASKS_FILE")
    command_edit

  elif [ -n "$MODE_NEW" ]; then
    command_new
    ##<%c
    num_tasks=$(grep -c -P "^\s*(${TASK_DONE_MARK}|\|\|)" "$PATH_TO_TASKS_FILE")
    ##c%>
    num_tasks_done=$(grep -c -P "^\s*${TASK_DONE_MARK}" "$PATH_TO_TASKS_FILE")

  else ## operações exclusivas para modo "action" (default)
    ##<%c
    num_tasks=$(grep -c -P "^\s*(${TASK_DONE_MARK}|\|\|)" "$PATH_TO_TASKS_FILE")
    ##c%>
    num_tasks_done=$(( num_tasks - num_items ))

    [ ${#created_dirs[@]} -ne 0 ] && {
      printf "~ Diretórios Criados em '${CURR_DIR,,}' (${#created_dirs[@]}):\\n"
      printf "%s\\n" "${created_dirs[@]}"
    }

    [ ${#created_files[@]} -ne 0 ] && {
      printf "~ Arquivos Criados em '${CURR_DIR,,}'   (${#created_files[@]}):\\n"
      printf "%s\\n" "${created_files[@]}"
    }

    [ ${#tasks_done[@]} -ne 0 ] && {
      confirm "~ ${#tasks_done[@]} Tarefa(s) Feita(s)" && {
        local emoji task_ref

        for real_line in "${tasks_done[@]}"; do
          sed -i "${real_line} s/^\s*||/${TASK_DONE_MARK} |/" "$PATH_TO_TASKS_FILE" && (( num_tasks_done++ ))

          ## referenciando o arquivo/diretório criado para esta tarefa
          if [ -n "${created_files[$real_line]+_}" ]; then
            sed -i "${real_line} s%$% [:${TASK_REF_EMOJIS[f]}:](${created_files[$real_line]})%" "$PATH_TO_TASKS_FILE"
          elif [ -n "${created_dirs[$real_line]+_}" ]; then
            sed -i "${real_line} s%$% [:${TASK_REF_EMOJIS[d]}:](${created_dirs[$real_line]})%" "$PATH_TO_TASKS_FILE"
          fi
        done
      }

      printf "\\n"
    }

    [ ${#tasks_to_remove[@]} -ne 0 ] && {
      confirm "~ ${#tasks_to_remove[@]} Tarefa(s) Removidas(s)" && {
        local lines_to_delete="${tasks_to_remove[@]/%/d;}" ## expressão única para apagar todas as linhas com o sed
        sed -i "$lines_to_delete" "$PATH_TO_TASKS_FILE" && num_tasks="$(( num_tasks - ${#tasks_to_remove[@]} ))"
      }

      printf "\\n"
    }

  fi


  percentage=$(( num_tasks_done*100 / num_tasks ))
  [ "$percentage" -eq 100 ] && color_key="g" ## se todas as tarefas foram concluídas

  sed -i -E "0,/(done-)[0-9]+(.+)\([0-9]+(.+of...)[0-9]+\)/ s//\1${percentage}\2(${num_tasks_done}\3${num_tasks})/" "$PATH_TO_TASKS_FILE" \
    && printf "%sNow %s${percentage}%% %s(${num_tasks_done} of ${num_tasks}) done!%s\\n" ${COLORS[w]} ${COLORS[$color_key]} ${COLORS[w]} ${COLORS[n]}

  exit 0 ## para sair do loop (na função main__) e fechar o programa
}


# ======================================================= #
# ==================== for items ======================== #
# @use: printf                                            #
# ======================================================= #

## Mostra o indicador de navegação.
## @args: [quantidade de linhas extras ocupadas pelo nome da tarefa]
print_navi() {
  move_to_navi $1
  printf "%s${NAVI_SYMBOL}%s" $NAVI_COLOR ${COLORS[n]}
  # move_to_column $column_sep ## desnecessário se o cursor estiver escondido
}

## Apaga o indicador de navegação e posiciona o cursor para sua coluna.
erase_navi() {
  move_to_navi
  printf "\\e[%dX" $NAVI_LENGTH ## escreve N brancos à direita
  move_to_navi
}

## Move o cursor para o indicador de navegação.
## @args: [quantidade de linhas extras ocupadas pelo nome da tarefa]
move_to_navi() {
  [ "${1:-0}" -gt 0 ] && move_up_lines $1
  move_to_column $NAVI_COLUMN
}


# ====================================================== #
# =============== items as tasks to edit =============== #
# @use: printf                                           #
# ====================================================== #

## Marca a tarefa, que está sobre o cursor, para futura edição.
mark_edit_task() {
  [ -n "${tasks_edit[$curr_item_index]+_}" ] && return ## verificando presença no array associativo
  local curr_item_value line_width voffset curr_item_index_normalized

  move_to_navi
  tasks_edit[$curr_item_index]="${list_items[$curr_item_index]%%:*}" ## associando o index com a linha real

  ##<%a
  curr_item_value="${list_items[$curr_item_index]#*:}"
  curr_item_index_normalized=$(( curr_item_index + 1 ))
  printf "%s%*d${SEPARATOR} %s${curr_item_value}%s" ${COLORS[p]} $offset $curr_item_index_normalized ${COLORS[gr]} ${COLORS[n]}

  line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value} ))
  voffset=$(( line_width / screen_width )) ## quantas linhas a mais o title está ocupando

  print_navi $voffset
  ##a%>
}

## Remove a marca "editar" da tarefa que está sobre o cursor.
remove_edit_mark() {
  [ -n "${tasks_edit[$curr_item_index]+_}" ] || return ## verificando presença no array associativo
  local curr_item_value line_width voffset curr_item_index_normalized

  move_to_navi
  unset tasks_edit[$curr_item_index] ## removendo do array associativo

  ##<%b
  curr_item_value="${list_items[$curr_item_index]#*:}"
  curr_item_index_normalized=$(( curr_item_index + 1 ))
  printf "%s%*d${SEPARATOR} ${curr_item_value}%s" ${COLORS[n]} $offset $curr_item_index_normalized ${COLORS[n]}

  line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value} ))
  voffset=$(( line_width / screen_width )) ## quantas linhas a mais o title está ocupando
  print_navi $voffset
  ##b%>
}


# ====================================================== #
# ================= items as tasks ===================== #
# @use: printf                                           #
# ====================================================== #

## Marca a tarefa, que está sobre o cursor, como "concluída".
mark_done_task() {
  [ -n "${tasks_done[$curr_item_index]+_}" ] && return ## verificando presença no array associativo
  local curr_item_value line_width voffset curr_item_index_normalized

  tasks_done[$curr_item_index]="${list_items[$curr_item_index]%%:*}" ## associando o index com a linha real
  move_to_navi
  unset tasks_to_remove[$curr_item_index] ## sobrescrevendo a ação 'remover tarefa'

  ##<%a
  curr_item_value="${list_items[$curr_item_index]#*:}"
  curr_item_index_normalized=$(( curr_item_index + 1 ))
  printf "%s%*d${SEPARATOR} %s${curr_item_value}%s" ${COLORS[g]} $offset $curr_item_index_normalized ${COLORS[gr]} ${COLORS[n]}

  line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value} ))
  voffset=$(( line_width / screen_width )) ## quantas linhas a mais o title está ocupando

  print_navi $voffset
  ##a%>
}

## Remove a marca "concluída" da tarefa que está sobre o cursor.
remove_done_mark() {
  [ -n "${tasks_done[$curr_item_index]+_}" ] || return ## verificando presença no array associativo
  local curr_item_value line_width voffset curr_item_index_normalized

  move_to_navi
  unset tasks_done[$curr_item_index] ## removendo do array associativo

  ##<%b
  curr_item_value="${list_items[$curr_item_index]#*:}"
  curr_item_index_normalized=$(( curr_item_index + 1 ))
  printf "%s%*d${SEPARATOR} ${curr_item_value}%s" ${COLORS[n]} $offset $curr_item_index_normalized ${COLORS[n]}

  line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value} ))
  voffset=$(( line_width / screen_width )) ## quantas linhas a mais o title está ocupando
  print_navi $voffset
  ##b%>
}

## Marca a tarefa, que está sobre o cursor, como "remover".
mark_delete_task() {
  local curr_item_value line_width voffset curr_item_index_normalized

  move_to_navi
  tasks_to_remove[$curr_item_index]="${list_items[$curr_item_index]%%:*}" ## associando o index com a linha real
  unset tasks_done[$curr_item_index] ## sobrescrevendo a ação 'marcar como feita'

  curr_item_value="${list_items[$curr_item_index]#*:}"
  curr_item_index_normalized=$(( curr_item_index + 1 ))
  printf "%s%*d${SEPARATOR} %s${curr_item_value}%s" ${COLORS[r]} $offset $curr_item_index_normalized ${COLORS[gr]} ${COLORS[n]}

  ##<%f
  line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value}+1 ))
  voffset=$(( line_width / screen_width )) ## quantas linhas a mais o title está ocupando
  print_navi $voffset
  ##f%>
}

## Remove a marca "remover" da tarefa que está sobre o cursor.
remove_delete_mark() {
  local curr_item_value line_width voffset curr_item_index_normalized

  move_to_navi
  unset tasks_to_remove[$curr_item_index] ## removendo do array associativo

  curr_item_value="${list_items[$curr_item_index]#*:}"
  curr_item_index_normalized=$(( curr_item_index + 1 ))
  printf "%s%*d${SEPARATOR} ${curr_item_value}%s" ${COLORS[n]} $offset $curr_item_index_normalized ${COLORS[n]}

  ##<%f
  line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value}+1 ))
  voffset=$(( line_width / screen_width )) ## quantas linhas a mais o title está ocupando
  print_navi $voffset
  ##f%>
}

## Marca ou desmarca uma tarefa para remoção.
toggle_delete_task() {
  if [ -n "${tasks_to_remove[$curr_item_index]+_}" ]; then remove_delete_mark; else mark_delete_task; fi
}

## Cria, um diretório para a tarefa corrente.
create_dir() {
  set_file_and_dir__ "${list_items[$curr_item_index]#*:}"
  [ -n "$normalized_task_name" ] || return 1

  mkdir -p "$dir" || return 1 ## ERROR
  emit_beep
  created_dirs[${list_items[$curr_item_index]%%:*}]=".${dir#${CURR_DIR,,}}"
}

## Abre o diretório da linguagem/projeto escolhida.
open_dir() {
  $EXPLORER "$CURR_DIR"
}

## Cria um arquivo de texto para a tarefa corrente.
## Após um sinal sonoro, espera a extensão do arquivo a ser criado.
create_file() {
  printf "\\e[?5h" ## turn on reverse video
  emit_beep
  read -rs extension
  printf "\\e[?5l" ## turn on normal video

  ## remover caracteres inválidos (uso das setas, brancos, etc)
  local extension="${extension//[[:cntrl:]]\[[[:alnum:]][[:punct:]]}"
  extension="${extension//[[:cntrl:]]\[[[:alnum:]]}"
  extension="${extension//[[:blank:]]}"
  extension="${extension#.}" ## remover o ponto se iniciar na 'extension'

  if [ -n "$extension" ]; then
    set_file_and_dir__ "${list_items[$curr_item_index]#*:}" "$extension"
    touch "$file" || return 1 ## ERROR
    emit_beep
    created_files[${list_items[$curr_item_index]%%:*}]=".${file#${CURR_DIR,,}}"
  fi
}

## Abre o link da tarefa corrente em um navegador.
## Um sinal sonoro será emitido em caso de sucesso.
open_title_link() {
  [ -n "$OPEN" ] || return 1
  local curr_task_line task_title

  curr_task_line=${list_items[$curr_item_index]%%:*}
  task_title="$(sed -En "$curr_task_line s/(http[^)]+).+/\\1/p" "$PATH_TO_TASKS_FILE")" ## recupera a primeira ocorrência

  [ -n "$task_title" ] && $OPEN "http${task_title#*\(http}" && emit_beep
}


# ===================================================== #
# ======================= items ======================= #
# @use: printf                                          #
# ===================================================== #

## Se possível, vai para o próximo item (desce).
next_item__() {
  erase_navi
  if [ $(( curr_item_index + 1 )) -ge "$num_items" ]; then
    ## vai para o primeiro item
    move_to_sof && curr_item_index=0
  else
    local curr_item_value="${list_items[$curr_item_index]#*:}"
    local curr_item_index_normalized=$(( curr_item_index + 1 ))
    local line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value} ))
    local voffset=$(( line_width / screen_width ))

    move_down_lines $(( voffset + 1))
    (( curr_item_index++ ))
  fi
  print_navi
}

## Se possível, volta para a o item anterior (sobe).
previous_item__() {
  erase_navi

  (( curr_item_index-- ))
  local curr_item_value="${list_items[$curr_item_index]#*:}"
  local curr_item_index_normalized=$(( curr_item_index + 2 ))
  local line_width=$(( offset + ${#curr_item_index_normalized} + ${#SEPARATOR} + ${#curr_item_value} ))
  local voffset=$(( line_width / screen_width ))

  if [ "$curr_item_index" -lt 0 ]; then
    ## vai para o último item
    curr_item_index=$(( num_items - 1))
    move_to_eof $(( voffset + 1 )) || curr_item_index=0
  else
    move_up_lines $(( voffset + 1 ))
  fi
  print_navi
}


# ===================================================== #
# ==================== generic ======================== #
# @use: printf read tput                                #
# ===================================================== #

move_to_sof() {
  printf "\\e[%sf" ${SOF_CURSOR_POS:-1}
}

## @args: [número de linhas a serem subtraídas da última]
move_to_eof() {
  [ -n "$NOT_MINGW_TERM" ] || return 1 ## ERROR (não compatível com o MinGW)
  local line="${eof_cursor_pos%;*}"
  printf "\\e[%d;1f" $(( line - ${1:-0} ))
}

erase_to_eol() {
  printf "\\e[K"
}

update_eof_cursor_pos__() {
  [ -n "$NOT_MINGW_TERM" ] || return 1 ## ERROR (não compatível com o MinGW)
  printf "\\e[6n" ## no MinGW não funciona como o esperado
  read -sdR eof_cursor_pos
  eof_cursor_pos="${eof_cursor_pos#*[}"
}

emit_beep() {
  printf "\007"
}

hide_cursor() {
  printf "\\e[?25l"
}

show_cursor() {
  printf "\\e[?25h"
}

clear_screen_exit() {
  ## limpar até o fim da tela (representando a "parada" do programa) e ativa o vídeo normal
  printf "\\n\\e[0J\\e[?5l"
  show_cursor
  exit 0
}

save_cursor() {
  printf "\\e[s"
}

restore_cursor() {
  printf "\\e[u"
}

## @args: [número da coluna]
move_to_column() {
  printf "\\e[%dG" ${1:-0}
}

## Move N linhas para baixo.
## Se N for 0, terá o mesmo efeito que se quando 1.
## @args: [número de linhas]
move_up_lines() {
  printf "\\e[%dA" ${1:-0}
}

## Move N linhas para baixo.
## Se N for 0, terá o mesmo efeito que se quando 1.
## @args: [número de linhas]
move_down_lines() {
  printf "\\e[%dB" ${1:-0}
}

## @args: [linhas] [colunas]
resize_window() {
  printf "\\e[8;${1};${2}t"
}

update_screen_width__() {
  screen_width=$(tput cols)
}


##############################
[ $# -lt 1 ] && show_help_exit
main__ "$@"
##############################

shopt -u extglob ## desativa ~ remover esta linha se for zsh
