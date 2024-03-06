%skeleton "lalr1.cc" // -*- C++ -*-
%require "3.8.1"
%header

%define api.token.raw

%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires {
#include "../parse_types.h"
class driver;
}

// The parsing context.
%param { driver& drv }

%locations

%define parse.trace
%define parse.error detailed
%define parse.lac full

%code {
#include "../driver.h"
#include <algorithm>
#define YYDEBUG 1

}



%define api.token.prefix {TOK_}
%token
  ASSIGN  "="
  TERNARY "?"
  COLON   ":"
  MINUS   "-"
  PLUS    "+"
  STAR    "*"
  SLASH   "/"
  AND     "&&"
  OR      "||"
  XOR     "^"
  NOT     "!"
  EQ      "=="
  NE      "!="
  LT      "<"
  LE      "<="
  GT      ">"
  GE      ">="
  LPAREN  "("
  RPAREN  ")"
  LBRACE  "{"
  RBRACE  "}"
  DOT     "."
  SEMICOLON ";"
  STARTP  "p["
  ENDP   "]"
  COMMA   ","

  CFG_START
  BND_START
  UPP_START
;

%token <std::string> IDENTIFIER 
%token <std::string> VARIABLE 
%token <std::string> ALIAS 
%token <float> FLOAT
%token <int> NUMBER 


%token <std::string> EXTERNALINPUT


%nterm <expr_ptr> exp

%nterm <node_attr_t> node_attribute
%nterm <node_attr_list_t> node_body


%nterm <std::vector<std::string>> nodes
%nterm <std::vector<int>> states


%printer { /* yyo << $$; */ } <*>;
%debug

%%
%start program;

program:
  CFG_START cfg_program
| BND_START bnd_program
| UPP_START upp_program

cfg_program:
  cfg_program cfg_declaration
| /* empty */

bnd_program:
  bnd_program bnd_declaration
| /* empty */

upp_program:
  upp_program upp_declaration
| %empty

cfg_declaration: 
  attr_declaration
| var_declaration
| const_declaration
| istate_declaration

attr_declaration: 
  IDENTIFIER "." IDENTIFIER "=" exp ";"     { drv.register_node_attribute(std::move($1), std::move($3), std::move($5)); }

var_declaration: 
  VARIABLE "=" exp ";"                      { drv.register_variable(std::move($1), std::move($3)); }

const_declaration: 
  IDENTIFIER "=" exp ";"                    { drv.register_constant(std::move($1), std::move($3)); }

istate_declaration: 
  "[" IDENTIFIER "]" "." IDENTIFIER "=" exp "[" NUMBER "]" "," exp "[" NUMBER "]" ";" {
                                                if ($5 != "istate") throw yy::parser::syntax_error(@5, "expected 'istate' keyword");
                                                if (!(($9 == 0 && $14 == 1) || ($9 == 1 && $14 == 0))) 
                                                    throw yy::parser::syntax_error(@9, "numbers in [] must be 0 and 1");
                                                drv.register_node_istate(std::move($2), std::move($7), std::move($12), $9);
                                            }


bnd_declaration:
  IDENTIFIER IDENTIFIER "{" node_body "}"   {
                                                std::transform($1.begin(), $1.end(), $1.begin(), ::tolower);
                                                if ($1 != "node") throw yy::parser::syntax_error(@1, "expected 'node' keyword");
                                                drv.register_node(std::move($2), std::move($4)); 
                                            }

node_body:
  node_body node_attribute                  { $1.push_back(std::move($2)); $$ = std::move($1); }
| node_attribute                            { $$.push_back(std::move($1)); }

node_attribute:
  IDENTIFIER "=" exp ";"                    { $$ = std::make_pair(std::move($1), std::move($3));}

upp_declaration:
  EXTERNALINPUT "=" exp ";"                 {drv.register_external_input(std::move($1),std::move($3));}     
| IDENTIFIER "=" exp ";"                    {drv.register_constant(std::move($1),std::move($3));}
| VARIABLE "=" exp ";"                      {drv.register_variable(std::move($1),std::move($3));}



%right "?";
%left XOR;
%left OR;
%left AND;
%left "==" "!=";
%left "<" "<=" ">" ">=";
%left "+" "-";
%left "*" "/";
%left UMINUS UPLUS NOT;
exp:
  FLOAT                                     { $$ = std::make_unique<literal_expression>($1); }
| NUMBER                                    { $$ = std::make_unique<literal_expression>($1); }
| IDENTIFIER                                { $$ = std::make_unique<identifier_expression>($1); }
| VARIABLE                                  { $$ = std::make_unique<variable_expression>($1); }
| ALIAS                                     { $$ = std::make_unique<alias_expression>($1); }
| exp "+" exp                               { $$ = std::make_unique<binary_expression>(operation::PLUS, std::move($1), std::move($3)); }
| exp "-" exp                               { $$ = std::make_unique<binary_expression>(operation::MINUS, std::move($1), std::move($3)); }
| exp "*" exp                               { $$ = std::make_unique<binary_expression>(operation::STAR, std::move($1), std::move($3)); }
| exp "/" exp                               { $$ = std::make_unique<binary_expression>(operation::SLASH, std::move($1), std::move($3)); }
| exp "==" exp                              { $$ = std::make_unique<binary_expression>(operation::EQ, std::move($1), std::move($3)); }
| exp "!=" exp                              { $$ = std::make_unique<binary_expression>(operation::NE, std::move($1), std::move($3)); }
| exp "<" exp                               { $$ = std::make_unique<binary_expression>(operation::LT, std::move($1), std::move($3)); }
| exp "<=" exp                              { $$ = std::make_unique<binary_expression>(operation::LE, std::move($1), std::move($3)); }
| exp ">" exp                               { $$ = std::make_unique<binary_expression>(operation::GT, std::move($1), std::move($3)); }
| exp ">=" exp                              { $$ = std::make_unique<binary_expression>(operation::GE, std::move($1), std::move($3)); }
| exp AND exp                               { $$ = std::make_unique<binary_expression>(operation::AND, std::move($1), std::move($3)); }
| exp OR exp                                { $$ = std::make_unique<binary_expression>(operation::OR, std::move($1), std::move($3)); }
| exp XOR exp                               { $$ = std::make_unique<binary_expression>(operation::XOR, std::move($1), std::move($3)); }
| exp "?" exp ":" exp %prec "?"             { $$ = std::make_unique<ternary_expression>(std::move($1), std::move($3), std::move($5)); }
| "(" exp ")"                               { $$ = std::make_unique<parenthesis_expression>(std::move($2)); }
| "-" exp %prec UMINUS                      { $$ = std::make_unique<unary_expression>(operation::MINUS, std::move($2)); }
| "+" exp %prec UPLUS                       { $$ = std::make_unique<unary_expression>(operation::PLUS, std::move($2)); }
| NOT exp                                   { $$ = std::make_unique<unary_expression>(operation::NOT, std::move($2)); }
| STARTP "(" nodes ")" "==" "(" states ")" ENDP     { $$ = std::make_unique<p_expression>(std::move($3), std::move($7));}
| EXTERNALINPUT                             {$$ = std::make_unique<external_input_expression>($1);}


states:
  states "," NUMBER                       { $1.push_back($3); $$ = std::move($1); }
| NUMBER                                  { $$.push_back(std::move($1)); }

nodes:
  nodes "," IDENTIFIER                       { $1.push_back(std::move($3)); $$ = std::move($1); }
| IDENTIFIER                                  {$$.push_back(std::move($1));}
%%

void
yy::parser::error (const location_type& l, const std::string& m)
{
  std::cerr << l << ": " << m << '\n';
}
