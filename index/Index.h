#ifndef INDEX_H
#define INDEX_H

#include <inttypes.h>
#include <functional>
#include <vector>
#include <unordered_set>
#include <memory>
#include <boost/any.hpp>
#include "structure/Element.h"

typedef std::function<int64_t(boost::any&)> HashFunction;
typedef std::function<bool(boost::any&, boost::any&)> EqualsFunction;

#define INDEX_MAX_LOAD_FACTOR 0.7
#define INDEX_SIZE_INCR 1000

class Index {
    private:
        HashFunction hash_func;
        EqualsFunction equals_func;
        std::vector<std::unordered_set<Element*>*> table;
        std::vector<boost::any> dict;
        
        size_t actual_size;

        /*
            Inserts the value/element pair into the Index.
        */
        void insert(std::vector<std::unordered_set<Element*>*>& table, std::vector<boost::any>& dict, Element* e, boost::any val) {
            int64_t hash = hash_func(val);
            size_t index = hash % dict.size();

            // if the slot was empty
            if(dict[index].empty()) {
                dict[index] = boost::any(val);
                table[index] = new std::unordered_set<Element*>;
                table[index]->insert(e);
                ++actual_size;
            } 

            // if the key was there and we just need to add the Element
            else if(equals_func(val, dict[index])) {
                table[index]->insert(e);
            }
            
            // if the slot was not empty
            else {
                while(!dict[index].empty() && !equals_func(dict[index], val)) {
                    ++index;
                    if(index >= dict.size()) index = 0;
                };

                // if the slot was empty
                if(dict[index].empty()) {
                    dict[index] = boost::any(val);
                    table[index] = new std::unordered_set<Element*>;
                    table[index]->insert(e);
                    ++actual_size;
                } 

                // if the key was there and we just need to add the Element
                else if(equals_func(val, dict[index])) {
                    table[index]->insert(e);
                }
            }
        }
    
    public:
        Index(HashFunction hash_func, EqualsFunction equals_func) {
            this->hash_func = hash_func;
            this->equals_func = equals_func;
            this->actual_size = 0;
            this->table.resize(INDEX_SIZE_INCR, nullptr);
            this->dict.resize(INDEX_SIZE_INCR);
        }

        bool is_indexed(boost::any& val) {
            int64_t hash = hash_func(val);
            size_t k = hash % dict.size();
            while(!dict[k].empty() && !equals_func(dict[k], val)) {
                ++k;
                if(k > dict.size()) k = 0;
            }
            
            if(dict[k].empty()) return false;
            return true; // index should never be full!
        }

        /*
            If necessary, rebuilds this index.
        */
        inline void rebuild() {
            if(static_cast<float>(actual_size) / static_cast<float>(table.size()) < INDEX_MAX_LOAD_FACTOR) return;

            size_t new_size = table.size()*5 + INDEX_SIZE_INCR;
            
            std::vector<std::unordered_set<Element*>*> new_table;
            new_table.resize(new_size, nullptr);

            std::vector<boost::any> new_dict;
            new_dict.resize(new_size);

            size_t old_size = table.size();
            for(int k = 0; k < old_size; ++k) if(!dict[k].empty()) {
                for(auto it = table[k]->begin(); it != table[k]->end(); ++it) {
                    insert(new_table, new_dict, *it, dict[k]);
                }
            }

            table.resize(new_size, nullptr);
            dict.resize(new_size);
            
            for(int k = 0; k < new_size; ++k) {
                if(table[k] != nullptr) delete table[k];

                table[k] = new_table[k];
                dict[k] = new_dict[k];
            }
        }

        /*
            Inserts the value/element pair into this Index.
        */
        void insert(Element* e, boost::any val) {
            insert(this->table, this->dict, e, val);
            rebuild();
        }

        std::unordered_set<Element*> get_elements(boost::any val) {
            size_t dict_size = dict.size();
            int64_t hash = hash_func(val);
            size_t index = hash % dict_size;
            
            std::unordered_set<Element*> s;

            if(dict[index].empty()) return s;

            while(!dict[index].empty() && !equals_func(dict[index], val)) {
                ++index;
                if(index >= dict_size) index = 0;
            }

            if(dict[index].empty()) return s;

            for(auto it = table[index]->begin(); it != table[index]->end(); ++it) s.insert(*it);
            return s;
        }

        /*
            Returns true if removed successfully, false if there was nothing
            to be removed.
        */
        bool remove(Element* e, boost::any val) {
            int64_t hash = hash_func(val);
            size_t index = hash % dict.size();

            // if the slot was empty
            if(dict[index].empty()) {
                return false;
            } 

            // if the key was there and we just need to delete the Element
            else if(equals_func(val, dict[index])) {
                table[index]->erase(e);
                if(table[index]->empty()) {
                    table[index] = nullptr;
                    dict[index] = boost::any();
                    --actual_size;
                }
            }
            
            // if the slot was not empty
            else {
                while(!dict[index].empty() && !equals_func(dict[index], val)) {
                    ++index;
                    if(index >= dict.size()) index = 0;
                };

                // if the slot was empty
                if(dict[index].empty()) {
                    return false;
                } 

                // if the key was there and we just need to delete the Element
                else if(equals_func(val, dict[index])) {
                    table[index]->erase(e);
                    if(table[index]->empty()) {
                        table[index] = nullptr;
                        dict[index] = boost::any();
                        --actual_size;
                    }
                }
            }

            rebuild();
            return true;
        }
};

#endif