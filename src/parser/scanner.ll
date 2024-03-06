%{ /* -*- C++ -*- */
# include <cerrno>
# include <climits>
# include <cfloat>
# include <cstdlib>
# include <cstring> // strerror
# include <string>
# include "../driver.h"
# include "parser.h"
%}

%option noyywrap nounput noinput batch debug

%{
  yy::parser::symbol_type make_NUMBER(const std::string &s, const yy::parser::location_type& loc);
  yy::parser::symbol_type make_FLOAT(const std::string &s, const yy::parser::location_type& loc);
  
  
  void syntax_error(const std::string &s, const yy::parser::location_type& loc);
%}

id                      [a-zA-Z][a-zA-Z_0-9]*
int                     [0-9]+
blank                   [ \t\r]
fracconst	              ([0-9]+\.[0-9]+)|([0-9]+\.)|(\.[0-9]+)
exppart		              [eE](\-|\+)?[0-9]+

%{
  // Code run each time a pattern is matched.
  # define YY_USER_ACTION  loc.columns (yyleng);
%}

%%

%{
  // A handy shortcut to the location held by the driver.
  yy::location& loc = drv.location;
  // Code run each time yylex is called.
  loc.step ();

  if (drv.start == driver::start_type::cfg)
  {
    drv.start = driver::start_type::none;
    return yy::parser::make_CFG_START(loc);
  }
  else if (drv.start == driver::start_type::bnd)
  {
    drv.start = driver::start_type::none;
    return yy::parser::make_BND_START(loc);
  }
  else if (drv.start == driver::start_type::upp)
  {
    drv.start = driver::start_type::none;
    return yy::parser::make_UPP_START(loc);
  }

%}

{blank}+                loc.step ();
"//".*                  loc.step ();
\n+                     loc.lines (yyleng); loc.step ();


"-"                     return yy::parser::make_MINUS(loc);
"+"                     return yy::parser::make_PLUS(loc);
"*"                     return yy::parser::make_STAR(loc);
"/"                     return yy::parser::make_SLASH(loc);
"("                     return yy::parser::make_LPAREN(loc);
")"                     return yy::parser::make_RPAREN(loc);
"="                     return yy::parser::make_ASSIGN(loc);
"?"                     return yy::parser::make_TERNARY(loc);
":"                     return yy::parser::make_COLON(loc);
("&&"|"&"|(?i:AND))     return yy::parser::make_AND(loc);
("||"|"|"|(?i:OR))      return yy::parser::make_OR(loc);
("^"|(?i:XOR))          return yy::parser::make_XOR(loc);
("!"|(?i:NOT))          return yy::parser::make_NOT(loc);
"=="                    return yy::parser::make_EQ(loc);
"!="                    return yy::parser::make_NE(loc);
"<"                     return yy::parser::make_LT(loc);
"<="                    return yy::parser::make_LE(loc);
">"                     return yy::parser::make_GT(loc);
">="                    return yy::parser::make_GE(loc);
"{"                     return yy::parser::make_LBRACE(loc);
"}"                     return yy::parser::make_RBRACE(loc);
"."                     return yy::parser::make_DOT(loc);
";"                     return yy::parser::make_SEMICOLON(loc);

"p\["                   return yy::parser::make_STARTP(loc);
"\]"                    return yy::parser::make_ENDP(loc);
","                     return yy::parser::make_COMMA(loc);



(?i:TRUE)               return yy::parser::make_NUMBER(1, loc);
(?i:FALSE)              return yy::parser::make_NUMBER(0, loc);

{int}                   return make_NUMBER(yytext, loc);

{fracconst}{exppart}? 	return make_FLOAT(yytext, loc);
[0-9]+{exppart}		      return make_FLOAT(yytext, loc);

{id}                    return yy::parser::make_IDENTIFIER(yytext, loc);
"$"{id}                 return yy::parser::make_VARIABLE(yytext, loc);
"@"{id}                 return yy::parser::make_ALIAS(yytext, loc);
"£"{id}                 return yy::parser::make_EXTERNALINPUT(yytext, loc);





.                       syntax_error(yytext, loc);
<<EOF>>                 return yy::parser::make_YYEOF(loc);

%%

yy::parser::symbol_type make_NUMBER(const std::string &s, const yy::parser::location_type &loc)
{
    errno = 0;
    long n = strtol(s.c_str(), nullptr, 10);
    if (!(INT_MIN <= n && n <= INT_MAX && errno != ERANGE))
        throw yy::parser::syntax_error(loc, "integer is out of range: " + s);
    return yy::parser::make_NUMBER((int)n, loc);
}

yy::parser::symbol_type make_FLOAT(const std::string &s, const yy::parser::location_type &loc)
{
    errno = 0;
    double n = strtod(s.c_str(), nullptr);
    if (errno == ERANGE)
        throw yy::parser::syntax_error(loc, "float is out of range: " + s);
    return yy::parser::make_FLOAT((float)n, loc);
}

void syntax_error(const std::string &s, const yy::parser::location_type &loc)
{
    throw yy::parser::syntax_error(loc, "invalid character: " + s);
}

void driver::scan_begin()
{
    yy_flex_debug = trace_scanning;
    if (file.empty() || file == "-")
        yyin = stdin;
    else if (!(yyin = fopen(file.c_str(), "r")))
    {
        std::cerr << "cannot open " << file << ": " << strerror(errno) << '\n';
        exit(EXIT_FAILURE);
    }
}

void driver::scan_end()
{
    fclose(yyin);
}
